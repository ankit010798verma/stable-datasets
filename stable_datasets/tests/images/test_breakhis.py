import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.breakhis import _IMAGE_RE, MAGNIFICATIONS, TEST_PATIENTS, TUMOR_TYPES, BreakHis
from stable_datasets.schema import Version


def test_breakhis_info():
    """Test dataset info, features, and source provenance without downloading."""
    builder = object.__new__(BreakHis)
    builder.__init__()
    info = builder.info

    assert info.description is not None
    assert set(info.features) == {"image", "label", "malignant", "magnification", "patient_id"}
    assert info.features["label"].names == TUMOR_TYPES
    assert info.features["label"].num_classes == 8
    assert info.features["malignant"].names == ["benign", "malignant"]
    assert info.features["magnification"].names == MAGNIFICATIONS
    assert BreakHis.VERSION == Version("1.0.0")
    assert builder.SOURCE["assets"]["train"].url == builder.SOURCE["assets"]["test"].url
    assert builder.SOURCE["assets"]["train"].checksum.startswith("sha256:")


def test_breakhis_path_parsing():
    """Archive paths parse into tumour type, magnification and patient ID; slide letters are dropped."""
    match = _IMAGE_RE.match(
        "BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-21998EF/400X/SOB_B_F-14-21998EF-400-013.png"
    )
    assert match is not None
    assert match["tumor_type"] == "fibroadenoma"
    assert match["magnification"] == "400X"
    assert match["patient_id"] == "14-21998"

    assert _IMAGE_RE.match("BreaKHis_v1/histology_slides/breast/benign/process_db_stat.py") is None
    assert len(TEST_PATIENTS) == 22


@pytest.mark.large
def test_breakhis_dataset():
    """Download BreakHis and check split sizes, sample format, and that no patient is in both splits."""
    ds_train = BreakHis(split="train")
    ds_test = BreakHis(split="test")
    assert len(ds_train) == 5555, f"Expected 5555 train samples, got {len(ds_train)}."
    assert len(ds_test) == 2354, f"Expected 2354 test samples, got {len(ds_test)}."

    sample = ds_test[0]
    expected_keys = {"image", "label", "malignant", "magnification", "patient_id"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    img = sample["image"]
    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}."
    img_np = np.array(img)
    assert img_np.ndim == 3 and img_np.shape[2] == 3, f"Expected RGB array, got shape {img_np.shape}."
    assert img_np.dtype == np.uint8, f"Expected uint8 dtype, got {img_np.dtype}."

    assert 0 <= sample["label"] < 8, f"Expected label in [0, 8), got {sample['label']}."
    assert sample["malignant"] == int(sample["label"] >= 4)
    assert 0 <= sample["magnification"] < 4

    train_patients = set(ds_train.table.column("patient_id").to_pylist())
    test_patients = set(ds_test.table.column("patient_id").to_pylist())
    assert test_patients == TEST_PATIENTS
    assert len(train_patients) == 48
    assert train_patients.isdisjoint(test_patients)

    ds_torch = ds_test.with_format("torch")
    torch_sample = ds_torch[0]
    assert torch_sample["image"].shape[0] == 3, f"Expected (3, H, W) torch tensor, got {torch_sample['image'].shape}."
