import pytest
from pathlib import Path
from ome_types.model import Image, Pixels, Channel, Plane, StageLabel

from ome2xarray.companion import sanitize_pixels, CompanionFile


def test_sanitize_pixels_basic():
    """Test basic sanitization with no stage label or multiple timepoints."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=2,
        size_c=2,
        size_t=1,
        channels=[
            Channel(id="Channel:0", name="GFP"),
            Channel(id="Channel:1", name="RFP")
        ],
        tiff_data_blocks=[],
        planes=[
            Plane(the_c=0, the_t=0, the_z=0, position_x=100.0, position_y=200.0, position_z=50.0),
            Plane(the_c=0, the_t=0, the_z=1, position_x=100.0, position_y=200.0, position_z=52.0),
        ]
    )

    image = Image(
        id="Image:0",
        name="test_image",
        pixels=pixels
    )

    sanitized = sanitize_pixels(image)

    # Check counts
    assert len(sanitized.tiff_data_blocks) == 4  # 2 channels * 1 time * 2 z
    assert len(sanitized.planes) == 4

    # Check file names (no stage, no time suffix)
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_image_w1GFP.ome.tif"
    assert sanitized.tiff_data_blocks[2].uuid.file_name == "test_image_w2RFP.ome.tif"

    # Check TiffData indices
    td0 = sanitized.tiff_data_blocks[0]
    assert td0.first_c == 0
    assert td0.first_t == 0
    assert td0.first_z == 0
    assert td0.ifd == 0

    td1 = sanitized.tiff_data_blocks[1]
    assert td1.first_c == 0
    assert td1.first_t == 0
    assert td1.first_z == 1
    assert td1.ifd == 1

    # Check planes
    plane0 = sanitized.planes[0]
    assert plane0.the_c == 0
    assert plane0.the_t == 0
    assert plane0.the_z == 0
    assert plane0.position_x == 100.0
    assert plane0.position_y == 200.0
    assert plane0.position_z == 50.0

    plane1 = sanitized.planes[1]
    assert plane1.position_z == 52.0


def test_sanitize_pixels_with_stage_label():
    """Test sanitization with stage label."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=1,
        size_t=1,
        channels=[Channel(id="Channel:0", name="DAPI")],
        tiff_data_blocks=[],
        planes=[
            Plane(the_c=0, the_t=0, the_z=0, position_x=0.0, position_y=0.0, position_z=10.0),
        ]
    )

    # Stage label with position 4 (should give _s5)
    image = Image(
        id="Image:0",
        name="test_4:Position5:0",
        pixels=pixels,
        stage_label=StageLabel(name="4:Position5:0")
    )

    sanitized = sanitize_pixels(image)

    assert len(sanitized.tiff_data_blocks) == 1
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_w1DAPI_s5.ome.tif"


def test_sanitize_pixels_with_multiple_timepoints():
    """Test sanitization with multiple timepoints."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=1,
        size_t=3,
        channels=[Channel(id="Channel:0", name="Brightfield")],
        tiff_data_blocks=[],
        planes=[
            Plane(the_c=0, the_t=0, the_z=0, position_x=0.0, position_y=0.0, position_z=0.0),
        ]
    )

    image = Image(
        id="Image:0",
        name="test",
        pixels=pixels
    )

    sanitized = sanitize_pixels(image)

    assert len(sanitized.tiff_data_blocks) == 3  # 1 channel * 3 times * 1 z
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_w1Brightfield_t1.ome.tif"
    assert sanitized.tiff_data_blocks[1].uuid.file_name == "test_w1Brightfield_t2.ome.tif"
    assert sanitized.tiff_data_blocks[2].uuid.file_name == "test_w1Brightfield_t3.ome.tif"
    
    # Assert planes exist and have correct time positions
    assert len(sanitized.planes) == 3
    assert sanitized.planes[0].the_t == 0
    assert sanitized.planes[1].the_t == 1
    assert sanitized.planes[2].the_t == 2


def test_sanitize_pixels_with_stage_and_time():
    """Test sanitization with both stage label and multiple timepoints."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=2,
        size_c=1,
        size_t=2,
        channels=[Channel(id="Channel:0", name="CFP")],
        tiff_data_blocks=[],
        planes=[
            Plane(the_c=0, the_t=0, the_z=0, position_x=5.0, position_y=10.0, position_z=20.0),
            Plane(the_c=0, the_t=0, the_z=1, position_x=5.0, position_y=10.0, position_z=22.0),
        ]
    )

    image = Image(
        id="Image:0",
        name="test_2:Pos3:1",
        pixels=pixels,
        stage_label=StageLabel(name="2:Pos3:1")
    )

    sanitized = sanitize_pixels(image)

    assert len(sanitized.tiff_data_blocks) == 4  # 1 channel * 2 times * 2 z
    # Should have both stage and time suffixes
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_w1CFP_s3_t1.ome.tif"
    assert sanitized.tiff_data_blocks[2].uuid.file_name == "test_w1CFP_s3_t2.ome.tif"


def test_sanitize_pixels_preserves_position_units():
    """Test that position units are preserved in sanitized planes."""
    from ome_types._autogenerated.ome_2016_06.units_length import UnitsLength
    
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=1,
        size_t=1,
        channels=[Channel(id="Channel:0", name="test")],
        tiff_data_blocks=[],
        planes=[
            Plane(
                the_c=0, the_t=0, the_z=0,
                position_x=100.0, position_y=200.0, position_z=300.0,
                position_x_unit=UnitsLength.MICROMETER,
                position_y_unit=UnitsLength.MICROMETER,
                position_z_unit=UnitsLength.MICROMETER
            ),
        ]
    )

    image = Image(id="Image:0", name="test", pixels=pixels)
    sanitized = sanitize_pixels(image)

    plane = sanitized.planes[0]
    assert plane.position_x_unit == UnitsLength.MICROMETER
    assert plane.position_y_unit == UnitsLength.MICROMETER
    assert plane.position_z_unit == UnitsLength.MICROMETER


def test_sanitize_pixels_no_planes():
    """Test sanitization when there are no existing planes."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=2,
        size_c=1,
        size_t=1,
        channels=[Channel(id="Channel:0", name="test")],
        tiff_data_blocks=[],
        planes=[]  # No existing planes
    )

    image = Image(id="Image:0", name="test", pixels=pixels)
    sanitized = sanitize_pixels(image)

    # Should still generate planes, but without position information
    assert len(sanitized.planes) == 2
    assert sanitized.planes[0].position_x is None
    assert sanitized.planes[0].position_y is None
    assert sanitized.planes[0].position_z is None


def test_companion_file_sanitize_image():
    """Test the CompanionFile.sanitize_image method."""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "20250910_VV7-0-0-6-ScanSlide"
        / "20250910_test4ch_2roi_3z_1_sg1.companion.ome"
    )

    companion_file = CompanionFile(companion_file_path)
    metadata = companion_file.get_ome_metadata()
    
    # Get original counts
    original_tiff_blocks = len(metadata.images[0].pixels.tiff_data_blocks)
    original_planes = len(metadata.images[0].pixels.planes)

    # Sanitize image 0
    companion_file.sanitize_image(0)

    # Get sanitized metadata
    sanitized_metadata = companion_file.get_ome_metadata()
    sanitized_pixels = sanitized_metadata.images[0].pixels

    # Should have the same counts (since we're regenerating)
    assert len(sanitized_pixels.tiff_data_blocks) == original_tiff_blocks
    assert len(sanitized_pixels.planes) == original_planes

    # Check that file names follow the expected pattern
    # Image name is "20250910_Test4ch_2ROI_3Z_1_0:Number1_sg:0"
    # Stage label is "0:Number1_sg:0"
    # So basename should be "20250910_Test4ch_2ROI_3Z_1"
    first_block = sanitized_pixels.tiff_data_blocks[0]
    assert first_block.uuid.file_name.startswith("20250910_Test4ch_2ROI_3Z_1_w1")


def test_companion_file_sanitize_image_invalid_index():
    """Test that sanitize_image raises IndexError for invalid index."""
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
        companion_file.sanitize_image(num_images)

    with pytest.raises(IndexError):
        companion_file.sanitize_image(-1)


def test_sanitize_pixels_removes_stage_label_from_image_name():
    """Test that stage label is removed from image.name to form basename."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=1,
        size_t=1,
        channels=[Channel(id="Channel:0", name="DAPI")],
        tiff_data_blocks=[],
        planes=[
            Plane(the_c=0, the_t=0, the_z=0, position_x=0.0, position_y=0.0, position_z=10.0),
        ]
    )

    # Example from the issue:
    # image.name: "20260109_SWI_3501_eft_col1_19:eft_20_sg:0"
    # stage_label.name: "19:eft_20_sg:0"
    # Expected basename: "20260109_SWI_3501_eft_col1"
    image = Image(
        id="Image:0",
        name="20260109_SWI_3501_eft_col1_19:eft_20_sg:0",
        pixels=pixels,
        stage_label=StageLabel(name="19:eft_20_sg:0")
    )

    sanitized = sanitize_pixels(image)

    assert len(sanitized.tiff_data_blocks) == 1
    # Should use basename without stage label
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "20260109_SWI_3501_eft_col1_w1DAPI_s20.ome.tif"


def test_sanitize_pixels_with_include_sg():
    """Test that include_sg=True adds _sg suffix at the right position."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=1,
        size_t=1,
        channels=[Channel(id="Channel:0", name="DAPI")],
        tiff_data_blocks=[],
        planes=[
            Plane(the_c=0, the_t=0, the_z=0, position_x=0.0, position_y=0.0, position_z=10.0),
        ]
    )

    image = Image(
        id="Image:0",
        name="test_0:Number1_sg:0",
        pixels=pixels,
        stage_label=StageLabel(name="0:Number1_sg:0")
    )

    sanitized = sanitize_pixels(image, include_sg=True)
    assert len(sanitized.tiff_data_blocks) == 1
    # Should include _sg1 after channel and before stage position
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_w1DAPI_sg1_s1.ome.tif"


@pytest.mark.parametrize("filename", [
    "20250910_test4ch_2roi_3z_1_sg1.companion.ome",
    "20250910_test4ch_2roi_3z_1_sg2.companion.ome"
])
def test_companion_file_filenames_with_and_without_sanitize_sg1(filename):
    """Test that filenames match between original and sanitized (with include_sg=True) for sg1 dataset."""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "20250910_VV7-0-0-6-ScanSlide"
        / filename
    )

    companion_file = CompanionFile(companion_file_path)
    metadata = companion_file.get_ome_metadata()
    
    # Get the first image
    original_image = metadata.images[0]
    original_filenames = {block.uuid.file_name for block in original_image.pixels.tiff_data_blocks}
    
    # Sanitize with include_sg=True
    companion_file.sanitize_image(0, include_sg=True)
    sanitized_metadata = companion_file.get_ome_metadata()
    sanitized_image = sanitized_metadata.images[0]
    sanitized_filenames = {block.uuid.file_name for block in sanitized_image.pixels.tiff_data_blocks}
    
    # The filenames should match
    assert original_filenames == sanitized_filenames, (
        f"Filenames don't match.\n"
        f"Original: {sorted(original_filenames)}\n"
        f"Sanitized: {sorted(sanitized_filenames)}"
    )


def test_companion_file_filenames_without_include_sg():
    """Test that filenames are different when sanitizing without include_sg for sg1/sg2 datasets."""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "20250910_VV7-0-0-6-ScanSlide"
        / "20250910_test4ch_2roi_3z_1_sg1.companion.ome"
    )

    companion_file = CompanionFile(companion_file_path)
    metadata = companion_file.get_ome_metadata()
    
    # Get the first image
    original_image = metadata.images[0]
    original_filenames = {block.uuid.file_name for block in original_image.pixels.tiff_data_blocks}
    
    # Sanitize without include_sg (default behavior)
    companion_file.sanitize_image(0, include_sg=False)
    sanitized_metadata = companion_file.get_ome_metadata()
    sanitized_image = sanitized_metadata.images[0]
    sanitized_filenames = {block.uuid.file_name for block in sanitized_image.pixels.tiff_data_blocks}
    
    # The filenames should NOT match (original has _sg1, sanitized without include_sg doesn't)
    assert original_filenames != sanitized_filenames, (
        "Filenames should be different when include_sg=False"
    )
    
    # Verify that original has _sg1 and sanitized doesn't
    for orig_fn in original_filenames:
        assert "_sg1_" in orig_fn, f"Original filename should contain _sg1_: {orig_fn}"
    
    for san_fn in sanitized_filenames:
        assert "_sg" not in san_fn, f"Sanitized filename should not contain _sg: {san_fn}"


def test_sanitize_pixels_with_channel_list():
    """Test sanitization with custom channel_list parameter."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=3,
        size_t=1,
        channels=[
            Channel(id="Channel:0", name="Red"),
            Channel(id="Channel:1", name="Green"),
            Channel(id="Channel:2", name="Blue")
        ],
        tiff_data_blocks=[],
        planes=[
            Plane(the_c=0, the_t=0, the_z=0, position_x=0.0, position_y=0.0, position_z=0.0),
        ]
    )

    image = Image(
        id="Image:0",
        name="test_image",
        pixels=pixels
    )

    # Sanitize with custom channel list (different order)
    channel_list = ["DAPI", "GFP"]
    sanitized = sanitize_pixels(image, channel_list=channel_list)

    # Check that the custom channel names are used
    assert len(sanitized.tiff_data_blocks) == 2
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_image_w1DAPI.ome.tif"
    assert sanitized.tiff_data_blocks[1].uuid.file_name == "test_image_w2GFP.ome.tif"


def test_sanitize_pixels_channel_list_none_uses_default():
    """Test that passing channel_list=None uses the default behavior."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=2,
        size_t=1,
        channels=[
            Channel(id="Channel:0", name="OriginalA"),
            Channel(id="Channel:1", name="OriginalB")
        ],
        tiff_data_blocks=[],
        planes=[]
    )

    image = Image(
        id="Image:0",
        name="test_image",
        pixels=pixels
    )

    # Sanitize with channel_list=None (explicit)
    sanitized = sanitize_pixels(image, channel_list=None)

    # Check that the original channel names are used
    assert len(sanitized.tiff_data_blocks) == 2
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_image_w1OriginalA.ome.tif"
    assert sanitized.tiff_data_blocks[1].uuid.file_name == "test_image_w2OriginalB.ome.tif"


def test_companion_file_sanitize_image_with_channel_list():
    """Test CompanionFile.sanitize_image with channel_list parameter."""
    companion_file_path = (
        Path(__file__).parent
        / "resources"
        / "20250910_VV7-0-0-6-ScanSlide"
        / "20250910_test4ch_2roi_3z_1_sg1.companion.ome"
    )

    companion_file = CompanionFile(companion_file_path)
    metadata = companion_file.get_ome_metadata()
    
    # Get original channel names for the first image
    original_image = metadata.images[0]
    original_channels = [ch.name for ch in original_image.pixels.channels]
    
    # Create a custom channel list with different names
    custom_channel_list = [f"Custom{i}" for i in range(len(original_channels))]
    
    # Sanitize with custom channel list
    companion_file.sanitize_image(0, channel_list=custom_channel_list)
    
    # Get sanitized metadata
    sanitized_metadata = companion_file.get_ome_metadata()
    sanitized_image = sanitized_metadata.images[0]
    
    # Check that custom channel names are used in filenames
    for block in sanitized_image.pixels.tiff_data_blocks:
        # The filename should contain one of the custom channel names
        assert any(custom_ch in block.uuid.file_name for custom_ch in custom_channel_list), (
            f"Filename {block.uuid.file_name} doesn't contain any custom channel name"
        )
