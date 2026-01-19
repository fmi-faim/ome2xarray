import pytest

from ome2xarray.companion import CompanionFile
from pathlib import Path


@pytest.mark.parametrize(
    "companion_file_name,image_index,expected_sum",
    [
        ("20250910_test4ch_2roi_3z_1_sg1.companion.ome", 6, 9404845159),
        ("20250910_test4ch_2roi_3z_1_sg2.companion.ome", 4, 8362360303),
    ],
)
def test_from_file_vv7(companion_file_name, image_index, expected_sum):
    folder = Path(__file__).parent / "resources" / "20250910_VV7-0-0-6-ScanSlide"
    companion_file_path = folder / companion_file_name

    assert companion_file_path.exists(), "companion.ome test file does not exist"

    companion_file = CompanionFile(companion_file_path)
    dataset = companion_file.get_dataset(image_index=image_index)
    assert dataset is not None
    # Check that all channel variables exist and have correct shape/dtype
    for ch_name, data_array in dataset.data_vars.items():
        assert data_array.shape == (1, 3, 512, 512)  # (T, Z, Y, X)
        assert data_array.dtype == "uint16"
        assert list(data_array.dims) == ["t", "z", "y", "x"]
    # Data integrity check (sum all channels)
    total_sum = sum(
        data_array.sum().compute() for data_array in dataset.data_vars.values()
    )
    assert total_sum == expected_sum


def test_get_dataset_invalid_image_index():
    """Test that get_dataset raises IndexError for invalid image_index"""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "20250910_VV7-0-0-6-ScanSlide"
        / "20250910_test4ch_2roi_3z_1_sg1.companion.ome"
    )
    companion_file = CompanionFile(companion_file_path)
    metadata = companion_file.get_ome_metadata()
    num_images = len(metadata.images)
    with pytest.raises(IndexError):
        companion_file.get_dataset(image_index=num_images)
    with pytest.raises(IndexError):
        companion_file.get_dataset(image_index=-1)


def test_get_datatree():
    """Test that get_datatree returns a DataTree with all images as children"""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "20250910_VV7-0-0-6-ScanSlide"
        / "20250910_test4ch_2roi_3z_1_sg1.companion.ome"
    )
    companion_file = CompanionFile(companion_file_path)
    with pytest.warns(UserWarning, match="Missing data: file"):
        datatree = companion_file.get_datatree()
    metadata = companion_file.get_ome_metadata()
    first_id = metadata.images[0].id

    # DataTree should contain at least one child node for the image
    assert datatree is not None
    assert first_id in datatree.children

    # First image should match what get_dataset returns
    ds_from_tree = datatree[first_id].ds
    with pytest.warns(UserWarning, match="Missing data: file"):
        ds_direct = companion_file.get_dataset(image_index=0)
    # Check all channel variables
    for ch_name in ds_direct.data_vars:
        arr_tree = ds_from_tree[ch_name]
        arr_direct = ds_direct[ch_name]
        assert arr_tree.shape == arr_direct.shape
        assert arr_tree.dtype == arr_direct.dtype


def test_get_datatree_single_image():
    """Test get_datatree for a file with multiple images"""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "20250910_VV7-0-0-6-ScanSlide"
        / "20250910_test4ch_2roi_3z_1_sg1.companion.ome"
    )
    companion_file = CompanionFile(companion_file_path)
    metadata = companion_file.get_ome_metadata()
    images = metadata.images
    expected_ids = [img.id for img in images]

    with pytest.warns(UserWarning, match="Missing data: file"):
        datatree = companion_file.get_datatree()

    # DataTree should have one child per image
    assert set(datatree.children.keys()) == set(expected_ids)

    # Verify the first image data
    # Assert warnings for missing data are raised
    first_id = expected_ids[0]
    ds = datatree[first_id].ds
    for ch_name, data_array in ds.data_vars.items():
        assert data_array.shape == (1, 3, 512, 512)  # (T, Z, Y, X)
        assert data_array.dtype == "uint16"
        assert data_array.sum().compute() == 0  # missing raw files


def test_z_positions_n2_lin28a633():
    """Test that z positions are correctly extracted from plane metadata"""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "n2-lin28a633_adult_2"
        / "n2-lin28a633_adult_2.companion.ome"
    )
    companion_file = CompanionFile(companion_file_path)
    with pytest.warns(UserWarning, match="Missing data: file"):
        dataset = companion_file.get_dataset(image_index=0)
    
    # Expected z positions: 65.4 to 84.0 in steps of 0.3
    import numpy as np
    expected_z = np.arange(65.4, 84.0 + 0.01, 0.3)  # +0.01 to include 84.0
    
    # Get z coordinates from dataset
    z_coords = dataset.coords["z"].values
    
    # Get size_z from metadata
    metadata = companion_file.get_ome_metadata()
    size_z = metadata.images[0].pixels.size_z
    
    assert len(z_coords) == size_z, f"Expected {size_z} z positions, got {len(z_coords)}"
    assert np.allclose(z_coords, expected_z, atol=1e-6), (
        f"Z positions don't match expected range. "
        f"Expected: {expected_z[:3]}...{expected_z[-3:]}, "
        f"Got: {z_coords[:3]}...{z_coords[-3:]}"
    )


@pytest.mark.parametrize(
    "companion_file_name,image_index,expected_sum",
    [
        ("20250910_test4ch_2roi_3z_1_sg1.companion.ome", 6, 9404845159),
    ],
)
def test_isolated_companion_file(
    companion_file_name, image_index, expected_sum
):
    folder = Path(__file__).parent / "resources" / "20250910_VV7-0-0-6-ScanSlide"

    main_companion_path = folder / companion_file_name
    isolated_companion_path = folder / "isolated" / companion_file_name

    main_companion = CompanionFile(main_companion_path)
    isolated_companion = CompanionFile(isolated_companion_path, data_folder=folder)

    main_dataset = main_companion.get_dataset(image_index=image_index)
    isolated_dataset = isolated_companion.get_dataset(image_index=image_index)

    # Check that datasets are identical
    for ch_name in main_dataset.data_vars:
        main_array = main_dataset[ch_name]
        isolated_array = isolated_dataset[ch_name]
        assert main_array.shape == isolated_array.shape
        assert main_array.dtype == isolated_array.dtype
        assert (main_array.compute() == isolated_array.compute()).all()

    # Data integrity check (sum all channels) on isolated dataset
    total_sum = sum(
        data_array.sum().compute() for data_array in isolated_dataset.data_vars.values()
    )
    assert total_sum == expected_sum
