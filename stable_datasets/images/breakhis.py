import io
import re
import tarfile

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_URL = "https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
_CHECKSUM = "sha256:d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640"

# Benign subtypes first, then malignant, so ``label < 4`` means benign.
TUMOR_TYPES = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
]
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# Patients held out for the test split. A patient is the numeric part of the slide ID
# (e.g. ``14-22549`` in ``SOB_B_A-14-22549AB-40-001.png``); slides lettered AB, CD, ... of the
# same patient are grouped, and three patients have slides of two tumour types, so this is
# 70 patients rather than the 82 slide folders in the archive. The list was chosen once to put
# ~30% of each tumour type's images in test with every type in both splits, then frozen.
TEST_PATIENTS = frozenset(
    {
        "14-9146",
        "14-9461",
        "14-10147",
        "14-10926",
        "14-11031",
        "14-11520",
        "14-12312",
        "14-12465",
        "14-13200",
        "14-13412",
        "14-13413",
        "14-13418",
        "14-14134",
        "14-15275",
        "14-15572",
        "14-16196",
        "14-16716",
        "14-17901",
        "14-18650",
        "14-19854",
        "14-22704",
        "14-29960",
    }
)

_IMAGE_RE = re.compile(
    r"^BreaKHis_v1/histology_slides/breast/(?:benign|malignant)/SOB/(?P<tumor_type>[a-z_]+)/[^/]+/(?P<magnification>\d+X)/"
    r"SOB_[BM]_[A-Z]+-(?P<patient_id>\d+-\d+)[A-Z]*-\d+-\d+\.png$"
)


class BreakHis(BaseDatasetBuilder):
    """Breast Cancer Histopathological Image Classification (BreakHis).

    BreakHis contains 7,909 microscopic images of breast tumour tissue (700x460 RGB PNG) collected from
    surgical biopsy slides stained with hematoxylin and eosin, at four magnifications (40X, 100X, 200X, 400X).
    Each image is labelled with one of eight tumour types, four benign (adenosis, fibroadenoma, phyllodes
    tumour, tubular adenoma) and four malignant (ductal, lobular, mucinous and papillary carcinoma).

    BreakHis has no official split, and images of one patient are near-duplicates of each other, so the splits
    here are patient-level (no patient appears in both):

    - train: 5,555 images from 48 patients
    - test: 2,354 images from 22 patients

    Each tumour type has 29-35% of its images in test, except adenosis (14%), which has only two patients.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@article{spanhol2016dataset,
            title={A Dataset for Breast Cancer Histopathological Image Classification},
            author={Spanhol, Fabio A and Oliveira, Luiz S and Petitjean, Caroline and Heutte, Laurent},
            journal={IEEE Transactions on Biomedical Engineering},
            volume={63},
            number={7},
            pages={1455--1462},
            year={2016},
            publisher={IEEE}
        }""",
        license="Free for research use with citation",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "BreakHis: 7,909 breast tumour histopathology images (700x460 RGB) from 82 slides of "
                "70 patients at 40X, 100X, 200X and 400X magnification, labelled with 8 tumour types "
                "(4 benign, 4 malignant). Patient-level 70/30 train/test split."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=TUMOR_TYPES),
                    "malignant": ClassLabel(names=["benign", "malignant"]),
                    "magnification": ClassLabel(names=MAGNIFICATIONS),
                    "patient_id": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        """Stream the PNGs of one split out of the archive, in archive order."""
        want_test = split == "test"
        idx = 0
        with tarfile.open(data_path, "r:gz") as tar:
            for member in tqdm(tar, desc=f"Processing {split} split"):
                match = _IMAGE_RE.match(member.name) if member.isfile() else None
                if match is None or (match["patient_id"] in TEST_PATIENTS) != want_test:
                    continue

                raw_bytes = tar.extractfile(member).read()
                with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
                    if pil_img.mode != "RGB":
                        buf = io.BytesIO()
                        pil_img.convert("RGB").save(buf, format="PNG")
                        raw_bytes = buf.getvalue()

                label = TUMOR_TYPES.index(match["tumor_type"])
                yield (
                    idx,
                    {
                        "image": raw_bytes,
                        "label": label,
                        "malignant": int(label >= 4),
                        "magnification": MAGNIFICATIONS.index(match["magnification"]),
                        "patient_id": match["patient_id"],
                    },
                )
                idx += 1
