from dask import delayed
from pathlib import Path
from ome_types import from_xml, OME

import dask.array as da
import numpy as np
import tifffile
import xarray as xr

class CompanionFile():

    _ome: OME
    _parent_path: Path

    def __init__(self, path: Path):
        with open(path, 'r', encoding="utf8") as file:
           self._parent_path = path.parent
           self._ome = from_xml(file.read())

    def get_dataarray(self):
        """
        Create a DataArray populated with the pixel data.
        """
        return create_ome_xarray(
            ome_metadata=self._ome,
            base_path=self._parent_path,
        )

    def get_ome_metadata(self) -> OME:
        """
        Get the OME metadata object.
        """
        return self._ome


class OMEReader:
    def __init__(self, ome_metadata, base_path):
        self.ome_metadata = ome_metadata
        self.base_path = Path(base_path)
        self.pixels = ome_metadata.images[0].pixels
        
        # Pre-load and cache all memory maps
        self.memmaps = {}
        self._load_memmaps()
        
        # Create spatial index for fast lookups
        self.block_index = {}
        self._create_block_index()
    
    def _load_memmaps(self):
        """Pre-load memory maps for all files"""
        file_names = {block.uuid.file_name for block in self.pixels.tiff_data_blocks}
        
        for file_name in file_names:
            file_path = self.base_path / file_name
            try:
                # Memory map the entire file once
                self.memmaps[file_name] = tifffile.memmap(file_path, mode='r')
            except Exception as e:
                print(f"Warning: Could not memory map {file_name}: {e}")
    
    def _create_block_index(self):
        """Create index mapping (t,c,z) -> (file_name, ifd)"""
        for block in self.pixels.tiff_data_blocks:
            key = (
                block.first_t, 
                block.first_c, 
                getattr(block, 'first_z', 0)
            )
            self.block_index[key] = (block.uuid.file_name, block.ifd)
    
    def read_plane(self, t, c, z):
        """Read a single plane - optimized for memory-mapped access"""
        key = (t, c, z)
        
        if key not in self.block_index:
            # Return zeros for missing planes
            return np.zeros(
                (self.pixels.size_y, self.pixels.size_x), 
                dtype=self.pixels.pixel_type
            )
        
        file_name, ifd = self.block_index[key]
        
        if file_name not in self.memmaps:
            # Fallback to direct file access if memmap failed
            file_path = self.base_path / file_name
            with tifffile.TiffFile(file_path) as tif:
                return tif.pages[ifd].asarray()
        
        # Use pre-loaded memory map
        return self.memmaps[file_name][ifd]

def create_ome_xarray(ome_metadata, base_path, chunks=None):
    """
    Create xarray DataArray from OME metadata with dask backing
    
    Parameters:
    -----------
    chunks : dict, optional
        Dask chunk sizes, e.g. {'t': 1, 'c': 1, 'z': 1, 'y': 2048, 'x': 2048}
        If None, defaults to single planes with full spatial dimensions
    """
    reader = OMEReader(ome_metadata, base_path)
    pixels = reader.pixels
    
    # Default chunking: one plane per chunk, full spatial dimensions
    if chunks is None:
        chunks = {'t': 1, 'c': 1, 'z': 1, 'y': pixels.size_y, 'x': pixels.size_x}
    
    # Create delayed reading function
    @delayed
    def read_plane_delayed(t, c, z):
        return reader.read_plane(t, c, z)
    
    # Build dask array by stacking delayed arrays
    # This approach scales well to thousands of planes
    arrays_by_t = []
    
    for t in range(pixels.size_t):
        arrays_by_c = []
        
        for c in range(pixels.size_c):
            arrays_by_z = []
            
            for z in range(pixels.size_z):
                delayed_plane = read_plane_delayed(t, c, z)
                
                dask_plane = da.from_delayed(
                    delayed_plane,
                    shape=(pixels.size_y, pixels.size_x),
                    dtype=pixels.type.value,
                )
                
                arrays_by_z.append(dask_plane)
            
            if arrays_by_z:
                arrays_by_c.append(da.stack(arrays_by_z, axis=0))
        
        if arrays_by_c:
            arrays_by_t.append(da.stack(arrays_by_c, axis=0))
    
    # Stack into final 5D array
    dask_array = da.stack(arrays_by_t, axis=0)
    
    # Rechunk according to specified chunks
    chunk_tuple = (
        chunks.get('t', 1),
        chunks.get('c', 1), 
        chunks.get('z', 1),
        chunks.get('y', pixels.size_y),
        chunks.get('x', pixels.size_x)
    )
    dask_array = dask_array.rechunk(chunk_tuple)
    
    # Create coordinate arrays
    coords = {
        't': np.arange(pixels.size_t),
        'c': np.arange(pixels.size_c),
        'z': np.arange(pixels.size_z), 
        'y': np.arange(pixels.size_y),
        'x': np.arange(pixels.size_x)
    }
    
    # Add metadata from OME if available
    attrs = {}
    if hasattr(pixels, 'physical_size_x') and pixels.physical_size_x:
        attrs['pixel_size_x'] = pixels.physical_size_x
    if hasattr(pixels, 'physical_size_y') and pixels.physical_size_y:
        attrs['pixel_size_y'] = pixels.physical_size_y
    if hasattr(pixels, 'physical_size_z') and pixels.physical_size_z:
        attrs['pixel_size_z'] = pixels.physical_size_z
    
    # Create xarray DataArray
    data_array = xr.DataArray(
        dask_array,
        dims=['t', 'c', 'z', 'y', 'x'],
        coords=coords,
        attrs=attrs,
        name='ome_image'
    )
    
    return data_array
