from dask import delayed
from pathlib import Path
from ome_types import from_xml, OME
from ome_types.model import Image, Pixels, TiffData, Plane
from uuid import uuid4

import dask.array as da
import numpy as np
import tifffile
import warnings
import xarray as xr


def sanitize_pixels(image: Image, *, include_sg: bool = False, channel_list: list[str] | None = None) -> Pixels:
    """
    Sanitize incomplete/corrupted pixels by regenerating tiff_data_blocks and planes.
    
    This function generates:
    - TiffData blocks for each (c, t, z) combination
    - File names based on the image name, channel info, stage position, and timepoint
    - Plane objects with position information repeated from existing planes
    
    Parameters:
    -----------
    image : Image
        The OME Image object containing pixels to sanitize
    include_sg : bool, optional
        If True, include stage group suffix (_sg1, _sg2, etc.) in file names.
        Default is False.
    channel_list : list[str] | None, optional
        List of channel names in the correct order. If provided, this list will be
        used instead of pixels.channels to determine channel names.
        Default is None (use pixels.channels).
        
    Returns:
    --------
    Pixels
        A copy of the pixels object with regenerated tiff_data_blocks and planes
    """
    pixels = image.pixels
    
    # Extract base name from image.name by removing stage_label suffix
    base_name = image.name
    if image.stage_label and image.stage_label.name:
        # Remove the stage_label suffix (including the preceding underscore)
        stage_label_suffix = f"_{image.stage_label.name}"
        if base_name.endswith(stage_label_suffix):
            base_name = base_name.removesuffix(stage_label_suffix)
    
    # Extract stage position number from stage_label if present
    stage_suffix = ""
    sg_suffix = ""
    if image.stage_label and image.stage_label.name:
        # Parse stage_label.name like "0:Number1_sg:0" or "4:Position5:0"
        # Extract the first number after splitting by ":"
        parts = image.stage_label.name.split(':')
        stage_pos = int(parts[0])
        stage_suffix = f"_s{stage_pos + 1}"
        
        # Extract stage group if include_sg is True
        if include_sg:
            # Split by last underscore to get the sg part
            # Example: "0:Number_1_sg:0" -> ["0:Number_1", "sg:0"]
            underscore_parts = image.stage_label.name.rsplit('_', 1)
            # Split by ':' to get the stage group index
            # Example: "sg:0" -> ["sg", "0"]
            sg_parts = underscore_parts[1].split(':')
            sg_index = int(sg_parts[1])
            sg_suffix = f"_sg{sg_index + 1}"
    
    # Collect position information from existing planes if available
    # We'll use the first plane's position as a template
    position_x = None
    position_y = None
    position_z_by_z = {}  # Map z index to position_z value
    position_x_unit = None
    position_y_unit = None
    position_z_unit = None
    
    if pixels.planes:
        # Get position from first plane
        first_plane = pixels.planes[0]
        position_x = first_plane.position_x
        position_y = first_plane.position_y
        position_x_unit = first_plane.position_x_unit
        position_y_unit = first_plane.position_y_unit
        position_z_unit = first_plane.position_z_unit
        
        # Collect z positions for each z index
        for plane in pixels.planes:
            if plane.the_z not in position_z_by_z and plane.position_z is not None:
                position_z_by_z[plane.the_z] = plane.position_z
    
    # Generate new TiffData blocks and Planes
    new_tiff_data_blocks = []
    new_planes = []
    
    # Track UUIDs per file name to ensure consistency
    file_uuids = {}

    if channel_list is None:
        channel_list = [channel.name for channel in pixels.channels]

    for c, channel_name in enumerate(channel_list):
        # Generate file name for this channel
        # Format: {base_name}_w{c+1}{channel_name}{sg_suffix}{stage_suffix}{time_suffix}.ome.tif
        file_base = f"{base_name}_w{c+1}{channel_name}{sg_suffix}{stage_suffix}"

        for t in range(pixels.size_t):
            # Add time suffix only if there are multiple timepoints
            time_suffix = f"_t{t+1}" if pixels.size_t > 1 else ""
            file_name = f"{file_base}{time_suffix}.ome.tif"

            # Generate a unique UUID for this file if not already done
            if file_name not in file_uuids:
                file_uuids[file_name] = f"urn:uuid:{uuid4()}"

            for z in range(pixels.size_z):
                # Create TiffData block
                tiff_data = TiffData(
                    first_c=c,
                    first_t=t,
                    first_z=z,
                    ifd=z,  # IFD index within the file
                    plane_count=1,
                    uuid=TiffData.UUID(
                        file_name=file_name,
                        value=file_uuids[file_name]
                    )
                )
                new_tiff_data_blocks.append(tiff_data)
                
                # Create Plane object - only include position units if they exist
                plane_kwargs = {
                    'the_c': c,
                    'the_t': t,
                    'the_z': z,
                    'position_x': position_x,
                    'position_y': position_y,
                    'position_z': position_z_by_z.get(z),
                }
                if position_x_unit is not None:
                    plane_kwargs['position_x_unit'] = position_x_unit
                if position_y_unit is not None:
                    plane_kwargs['position_y_unit'] = position_y_unit
                if position_z_unit is not None:
                    plane_kwargs['position_z_unit'] = position_z_unit
                
                plane = Plane(**plane_kwargs)
                new_planes.append(plane)
    
    # Create a copy of pixels with new tiff_data_blocks and planes
    pixels_dict = pixels.model_dump()
    pixels_dict['tiff_data_blocks'] = new_tiff_data_blocks
    pixels_dict['planes'] = new_planes
    
    return Pixels(**pixels_dict)


class CompanionFile:
    _ome: OME
    _data_folder: Path
    _path: Path

    def __init__(self, path: Path, data_folder: Path | None = None):
        self._path = path
        with open(path, "r", encoding="utf8") as file:
            self._data_folder = data_folder if data_folder is not None else path.parent
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
            base_path=self._data_folder,
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

    def sanitize_image(self, image_index: int, include_sg: bool = False, channel_list: list[str] | None = None) -> None:
        """
        Sanitize an image's pixels by regenerating tiff_data_blocks and planes.
        
        This is useful for incomplete/corrupted companion.ome files where the
        tiff_data_blocks or planes are missing or incorrect.
        
        Parameters:
        -----------
        image_index : int
            Index of the image to sanitize
        include_sg : bool, optional
            If True, include stage group suffix (_sg1, _sg2, etc.) in file names.
            Default is False.
        channel_list : list[str] | None, optional
            List of channel names in the correct order. If provided, this list will be
            used instead of pixels.channels to determine channel names. The length must
            match pixels.size_c. Default is None (use pixels.channels).
        """
        if image_index < 0 or image_index >= len(self._ome.images):
            raise IndexError(
                f"image_index {image_index} out of range. CompanionFile contains {len(self._ome.images)} image(s)."
            )
        
        image = self._ome.images[image_index]
        
        # Sanitize the pixels
        sanitized_pixels = sanitize_pixels(image, include_sg=include_sg, channel_list=channel_list)
        
        # Replace the image's pixels with sanitized version
        # We need to create a new image with the sanitized pixels
        image_dict = image.model_dump()
        image_dict['pixels'] = sanitized_pixels
        self._ome.images[image_index] = Image(**image_dict)


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

    # Per-plane positions (z, y, x) from OME metadata
    # Create mapping from the_z index to position_z value
    z_position_map = {}
    for plane in pixels.planes:
        if plane.the_z not in z_position_map:
            z_position_map[plane.the_z] = plane.position_z
        elif not np.isclose(z_position_map[plane.the_z], plane.position_z):
            raise ValueError(
                f"Inconsistent position_z values for the_z={plane.the_z}: "
                f"{z_position_map[plane.the_z]} != {plane.position_z}"
            )
    
    # Build z_positions array using the mapping
    z_positions = [z_position_map[z] for z in range(pixels.size_z)]
    
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
