import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.chest_xray import ChestXRay
from stable_datasets.schema import Version


def test_chest_xray_info():
    """Test dataset info, features, and source provenance without downloading."""
    builder = object.__new__(ChestXRay)
    builder.__init__()
    info = builder.info

    assert info.description is not None
    assert "image" in info.features
    assert "label" in info.features
    assert info.features["label"].names == ["NORMAL", "PNEUMONIA"]
    assert info.features["label"].num_classes == 2
    assert ChestXRay.VERSION == Version("1.0.0")
    assert builder.SOURCE["homepage"] == "https://data.mendeley.com/datasets/rscbjbr9sj/3"
    assert builder.SOURCE["license"] == "CC BY 4.0"


@pytest.mark.large
def test_chest_xray_dataset():
    """Download and test validation split of Chest X-Ray."""
    ds_val = ChestXRay(split="validation")
    assert len(ds_val) == 16, f"Expected 16 validation samples, got {len(ds_val)}."

    sample = ds_val[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    img = sample["image"]
    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}."
    img_np = np.array(img)
    assert img_np.ndim == 3, f"Expected 3D RGB array, got shape {img_np.shape}."
    assert img_np.shape[2] == 3, f"Expected 3 channels (RGB), got {img_np.shape[2]}."
    assert img_np.dtype == np.uint8, f"Expected uint8 dtype, got {img_np.dtype}."

    label = sample["label"]
    assert isinstance(label, int), f"Expected int label, got {type(label)}."
    assert label in (0, 1), f"Expected label in (0, 1), got {label}."

    ds_torch = ds_val.with_format("torch")
    torch_sample = ds_torch[0]
    assert "image" in torch_sample
    assert torch_sample["image"].shape[0] == 3, f"Expected (3, H, W) torch tensor, got {torch_sample['image'].shape}."
