import pytest

from ome_dataarray.companion import CompanionFile
from pathlib import Path

@pytest.fixture
def companion_file_path():
    return Path(__file__).parent / "resources" / "20250612_6pp7_1_gain50_300ms_488_10_640_2_10.companion.ome"


def test_from_file(companion_file_path):
    assert companion_file_path.exists(), "companion.ome test file does not exist"

    companion_file = CompanionFile(companion_file_path)
    data_array = companion_file.get_dataarray()
    assert data_array is not None
    assert data_array.shape == (100, 3, 1, 512, 512)  # (T, C, Z, Y, X)
    assert data_array.dtype == 'uint16'
