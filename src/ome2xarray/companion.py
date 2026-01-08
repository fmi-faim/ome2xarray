from dask import delayed
from pathlib import Path
from ome_types import from_xml, OME
from ome_types.model import Image

import dask.array as da
import numpy as np
import tifffile
import warnings
import xarray as xr


class CompanionFile:
    _ome: OME
    _parent_path: Path

    def __init__(self, path: Path):
        with open(path, "r", encoding="utf8") as file:
            self._parent_path = path.parent
            self._ome = from_xml(file.read())

    def get_dataset(self, image_index: int) -> xr.Dataset:
        """
        Create a Dataset for one image/series from the companion.ome file.
        Channels are included as separate DataArrays with dims (t, z, y, x).

        Parameters:
        -----------
        image_index : int
            Index of the image/series to retrieve
        Returns:
        --------
        xr.Dataset
            Dataset containing a DataArray per channel with dims (t, z, y, x).
        """
        if image_index < 0 or image_index >= len(self._ome.images):
            raise IndexError(
                f"image_index {image_index} out of range. CompanionFile contains {len(self._ome.images)} image(s)."
            )
        return _create_channel_dataset(
            image=self._ome.images[image_index],
            base_path=self._parent_path,
        )

    def get_datatree(self) -> xr.DataTree:
        """
        Create an xarray.DataTree containing all images/series from the companion.ome file.

        Each image is included as a child node, with its own Dataset (containing the channel DataArrays and coordinates).

        Returns:
        --------
        DataTree
            xarray.DataTree with each image as a child node, each node containing a Dataset with the channel DataArrays and its coordinates.
        """
        children = {}
        for idx, image in enumerate(self._ome.images):
            ds = self.get_dataset(idx)
            children[image.id] = xr.DataTree(dataset=ds, name=image.id)
        return xr.DataTree(name="root", children=children)

    def get_ome_metadata(self) -> OME:
        """
        Get the OME metadata object.
        """
        return self._ome


def _create_channel_dataset(image: Image, base_path, chunks=None):
    """
    Build an xarray.Dataset for one OME Image where each channel is a
    separate DataArray with dims (t, z, y, x). Time and spatial coordinates
    are shared across variables.
    """
    reader = OMEImageReader(image, base_path)
    pixels = reader.pixels

    if chunks is None:
        chunks = {"t": 1, "z": 1, "y": pixels.size_y, "x": pixels.size_x}

    # Per-plane positions (z, y, x) from OME metadata, fallback to pixel indices * pixel size
    z_positions = [
        plane.position_z if plane.position_z is not None else 0.0
        for plane in pixels.planes[: pixels.size_z]
    ]
    x_pixel_size = pixels.physical_size_x or 0.0
    y_pixel_size = pixels.physical_size_y or 0.0
    x_offsets = [(plane.position_x or 0.0) for plane in pixels.planes[: pixels.size_z]]
    y_offsets = [(plane.position_y or 0.0) for plane in pixels.planes[: pixels.size_z]]
    if not all(np.isclose(x_offsets[0], xo) for xo in x_offsets):
        raise ValueError(
            "position_x offset is not the same across all planes; cannot create 1D calibrated x coordinate."
        )
    if not all(np.isclose(y_offsets[0], yo) for yo in y_offsets):
        raise ValueError(
            "position_y offset is not the same across all planes; cannot create 1D calibrated y coordinate."
        )
    x_offset = x_offsets[0]
    y_offset = y_offsets[0]
    x_coords = np.arange(pixels.size_x) * x_pixel_size + x_offset
    y_coords = np.arange(pixels.size_y) * y_pixel_size + y_offset

    coords = {
        "t": np.arange(pixels.size_t),
        "z": z_positions,
        "y": y_coords,
        "x": x_coords,
    }

    attrs = {
        "pixel_size_x": pixels.physical_size_x,
        "pixel_size_y": pixels.physical_size_y,
        "pixel_size_z": pixels.physical_size_z,
    }

    channel_names = [ch.name for ch in pixels.channels]
    data_vars = {}
    # Outer loop over channels
    for c, ch_name in enumerate(channel_names):
        # Build dask array for this channel: shape (t, z, y, x)
        arrays_by_t = []
        for t in range(pixels.size_t):
            arrays_by_z = []
            for z in range(pixels.size_z):

                @delayed
                def read_plane_delayed(t=t, c=c, z=z):
                    return reader.read_plane(c, t, z)

                dask_plane = da.from_delayed(
                    read_plane_delayed(),
                    shape=(pixels.size_y, pixels.size_x),
                    dtype=pixels.type.value,
                )
                arrays_by_z.append(dask_plane)
            arrays_by_t.append(da.stack(arrays_by_z, axis=0))
        dask_array = da.stack(arrays_by_t, axis=0)
        chunk_tuple = (
            chunks.get("t", 1),
            chunks.get("z", 1),
            chunks.get("y", pixels.size_y),
            chunks.get("x", pixels.size_x),
        )
        dask_array = dask_array.rechunk(chunk_tuple)
        data_vars[ch_name] = xr.DataArray(
            dask_array,
            dims=["t", "z", "y", "x"],
            coords=coords,
            attrs=attrs,
            name=ch_name,
        )
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


class OMEImageReader:
    """Reader for a single OME Image (series)."""

    def __init__(self, image: Image, base_path):
        self.image = image
        self.base_path = Path(base_path)
        self.pixels = image.pixels

        # Create spatial index for fast lookups
        self.block_index = {}
        self._create_block_index()

    # Files are opened lazily when a plane is read to keep behavior simple
    # and rely on delayed Dask tasks for loading TIFF plane data.

    def _create_block_index(self):
        """Create index mapping (c,t,z) -> (file_name, ifd)"""
        for block in self.pixels.tiff_data_blocks:
            key = (block.first_c, block.first_t, getattr(block, "first_z", 0))
            if (self.base_path / block.uuid.file_name).exists():
                self.block_index[key] = (block.uuid.file_name, block.ifd)
            else:
                msg = f"Missing data: file {block.uuid.file_name} not found in {self.base_path}."
                warnings.warn(msg, UserWarning)


    def read_plane(self, c, t, z):
        """Read a single (c, t, z) plane by opening the TIFF file on demand."""
        key = (c, t, z)

        if key not in self.block_index:
            # Return zeros for missing planes
            return np.zeros(
                (self.pixels.size_y, self.pixels.size_x), dtype=self.pixels.type.value
            )

        file_name, ifd = self.block_index[key]

        file_path = self.base_path / file_name
        with tifffile.TiffFile(file_path) as tif:
            return tif.pages[ifd].asarray()
