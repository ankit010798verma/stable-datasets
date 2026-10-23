import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import PatternNet
from stable_datasets.images.patternnet import CLASS_NAMES
from stable_datasets.schema import Version


def test_patternnet_info():
    """Test dataset info, features, and source provenance without downloading."""
    builder = object.__new__(PatternNet)
    builder.__init__()
    info = builder.info

    assert info.description is not None
    assert "image" in info.features
    assert "label" in info.features
    assert info.features["label"].names == CLASS_NAMES
    assert info.features["label"].num_classes == 38
    assert PatternNet.VERSION == Version("1.0.0")
    assert builder.SOURCE["homepage"] == "https://sites.google.com/view/zhouwx/dataset"


@pytest.mark.large
def test_patternnet_dataset():
    """Download and test loading of PatternNet test split."""
    ds_test = PatternNet(split="test")
    assert len(ds_test) == 6080, f"Expected 6080 test samples, got {len(ds_test)}."

    sample = ds_test[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    img = sample["image"]
    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}."
    img_np = np.array(img)
    assert img_np.shape == (256, 256, 3), f"Expected shape (256, 256, 3), got {img_np.shape}."
    assert img_np.dtype == np.uint8, f"Expected uint8 dtype, got {img_np.dtype}."

    label = sample["label"]
    assert isinstance(label, int), f"Expected int label, got {type(label)}."
    assert 0 <= label < 38, f"Expected label in [0, 37], got {label}."

    ds_torch = ds_test.with_format("torch")
    torch_sample = ds_torch[0]
    assert "image" in torch_sample
    assert torch_sample["image"].shape == (3, 256, 256), f"Expected (3, 256, 256) tensor, got {torch_sample['image'].shape}."
