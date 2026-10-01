import h5py
import numpy as np
import pytest
import torch

from the_well.data.datasets import DeltaWellDataset, WellDataset
from the_well.utils.dummy_data import write_dummy_data


@pytest.fixture
def data_dir(tmp_path):
    """Dummy data (32 x 32) plus two constant fields that vary along one axis only
    (stored with size 1 along the other)."""
    path = tmp_path / "data"
    path.mkdir()
    write_dummy_data(path / "dummy.hdf5")
    with h5py.File(path / "dummy.hdf5", "r+") as file:
        group = file["t0_fields"]
        for name, use_dims, shape in [
            ("x_profile", [True, False], (32, 1)),
            ("y_profile", [False, True], (1, 32)),
        ]:
            dset = group.create_dataset(name, data=np.random.rand(*shape).astype(np.float32))
            dset.attrs["dim_varying"] = use_dims
            dset.attrs["sample_varying"] = False
            dset.attrs["time_varying"] = False
        group.attrs["field_names"] = ["constant_field", "x_profile", "y_profile"]
    return path


CUT_TO_SLAB = ["input_fields", "output_fields", "constant_fields", "space_grid"]


@pytest.mark.parametrize("dataset_cls", [WellDataset, DeltaWellDataset])
@pytest.mark.parametrize("slab", [(0, 8, 24), (1, 0, 16), (0, 31, 32)])
def test_slab_matches_full_read(data_dir, dataset_cls, slab):
    full = dataset_cls(path=str(data_dir), n_steps_input=2, n_steps_output=3)
    part = dataset_cls(path=str(data_dir), n_steps_input=2, n_steps_output=3, slab=slab)
    axis, start, stop = slab
    for index in [0, len(full) - 1]:
        whole, cut = full[index], part[index]
        assert whole.keys() == cut.keys()
        for key in whole:
            expected = whole[key]
            if key in CUT_TO_SLAB:
                # Spatial axes follow the time axis, except in the space grid
                dim = axis if key in ("constant_fields", "space_grid") else axis + 1
                expected = expected.narrow(dim, start, stop - start)
            torch.testing.assert_close(cut[key], expected, rtol=0, atol=0)
    # Metadata still describes the full domain
    assert part.metadata.spatial_resolution == (32, 32)


def test_changing_slab_does_not_reuse_cached_fields(data_dir):
    dataset = WellDataset(path=str(data_dir), slab=(0, 0, 8))
    first = dataset[0]
    dataset.slab = (0, 8, 16)
    second = dataset[0]
    full = WellDataset(path=str(data_dir))[0]
    for key in ("constant_fields", "space_grid"):
        torch.testing.assert_close(first[key], full[key][0:8], rtol=0, atol=0)
        torch.testing.assert_close(second[key], full[key][8:16], rtol=0, atol=0)


@pytest.mark.parametrize("slab", [(2, 0, 8), (0, 0, 33), (0, 8, 8)])
def test_invalid_slab_raises(data_dir, slab):
    with pytest.raises(ValueError):
        WellDataset(path=str(data_dir), slab=slab)
