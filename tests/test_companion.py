import pytest

from ome_dataarray.companion import CompanionFile
from pathlib import Path


@pytest.mark.parametrize(
    "companion_file_name,expected_sum",
    [
        ("20250910_test4ch_2roi_3z_1_sg1.companion.ome", 9061879471),
        ("20250910_test4ch_2roi_3z_1_sg2.companion.ome", 10522218363),
    ],
)
def test_from_file_vv7(companion_file_name, expected_sum):
    folder = Path(__file__).parent / "resources" / "20250910_VV7-0-0-6-ScanSlide"
    companion_file_path = folder / companion_file_name

    assert companion_file_path.exists(), "companion.ome test file does not exist"

    companion_file = CompanionFile(companion_file_path)
    data_array = companion_file.get_dataarray()
    assert data_array is not None
    assert data_array.shape == (1, 4, 3, 512, 512)  # (T, C, Z, Y, X)
    assert data_array.dtype == 'uint16'
    assert list(data_array.dims) == ["t", "c", "z", "y", "x"]
    # Data integrity check
    assert data_array.sum().compute() == expected_sum
