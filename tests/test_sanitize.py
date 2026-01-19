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

    sanitized = sanitize_pixels(image, "dataset.companion.ome")

    # Check counts
    assert len(sanitized.tiff_data_blocks) == 4  # 2 channels * 1 time * 2 z
    assert len(sanitized.planes) == 4

    # Check file names (no stage, no time suffix)
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "dataset_w1GFP.ome.tif"
    assert sanitized.tiff_data_blocks[2].uuid.file_name == "dataset_w2RFP.ome.tif"

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
        name="test",
        pixels=pixels,
        stage_label=StageLabel(name="4:Position5:0")
    )

    sanitized = sanitize_pixels(image, "my_experiment.companion.ome")

    assert len(sanitized.tiff_data_blocks) == 1
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "my_experiment_w1DAPI_s5.ome.tif"


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

    sanitized = sanitize_pixels(image, "timelapse.companion.ome")

    assert len(sanitized.tiff_data_blocks) == 3  # 1 channel * 3 times * 1 z
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "timelapse_w1Brightfield_t1.ome.tif"
    assert sanitized.tiff_data_blocks[1].uuid.file_name == "timelapse_w1Brightfield_t2.ome.tif"
    assert sanitized.tiff_data_blocks[2].uuid.file_name == "timelapse_w1Brightfield_t3.ome.tif"
    
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
        name="test",
        pixels=pixels,
        stage_label=StageLabel(name="2:Pos3:1")
    )

    sanitized = sanitize_pixels(image, "complex.companion.ome")

    assert len(sanitized.tiff_data_blocks) == 4  # 1 channel * 2 times * 2 z
    # Should have both stage and time suffixes
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "complex_w1CFP_s3_t1.ome.tif"
    assert sanitized.tiff_data_blocks[2].uuid.file_name == "complex_w1CFP_s3_t2.ome.tif"


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
    sanitized = sanitize_pixels(image, "test.companion.ome")

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
    sanitized = sanitize_pixels(image, "test.companion.ome")

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
    first_block = sanitized_pixels.tiff_data_blocks[0]
    assert first_block.uuid.file_name.startswith("20250910_test4ch_2roi_3z_1_sg1_w1")


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


def test_sanitize_pixels_channel_without_name():
    """Test sanitization with a channel that has no name."""
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        type="uint16",
        size_x=512,
        size_y=512,
        size_z=1,
        size_c=1,
        size_t=1,
        channels=[Channel(id="Channel:0", name=None)],  # No name
        tiff_data_blocks=[],
        planes=[]
    )

    image = Image(id="Image:0", name="test", pixels=pixels)
    sanitized = sanitize_pixels(image, "test.companion.ome")

    # Should generate a default channel name
    assert sanitized.tiff_data_blocks[0].uuid.file_name == "test_w1Channel0.ome.tif"
