import io

import pyarrow.parquet as pq
from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


CLASS_NAMES = [
    "airplane",
    "baseball field",
    "basketball court",
    "beach",
    "bridge",
    "cemetery",
    "chaparral",
    "christmas tree farm",
    "closed road",
    "coastal mansion",
    "crosswalk",
    "dense residential",
    "ferry terminal",
    "football field",
    "forest",
    "freeway",
    "golf course",
    "harbor",
    "intersection",
    "mobile home park",
    "nursing home",
    "oil gas field",
    "oil well",
    "overpass",
    "parking lot",
    "parking space",
    "railway",
    "river",
    "runway",
    "runway marking",
    "shipping yard",
    "solar panel",
    "sparse residential",
    "storage tank",
    "swimming pool",
    "tennis court",
    "transformer station",
    "wastewater treatment plant",
]


class PatternNet(BaseDatasetBuilder):
    """
    PatternNet Remote Sensing Image Classification Dataset.

    PatternNet is a large-scale high-resolution remote sensing dataset collected for
    performance evaluation of remote sensing image retrieval and scene classification.
    Images are collected from Google Earth or the Google Maps API.

    The dataset contains 38 classes with 800 images per class (30,400 images total).
    Each image is 256x256 pixels in RGB format.

    Splits (standard 80/20 train/test split):

    - train: 24,320 images (640 per class)
    - test: 6,080 images (160 per class)
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://sites.google.com/view/zhouwx/dataset",
        assets={
            f"train_{i}": DownloadInfo(
                url=f"https://huggingface.co/datasets/blanchon/PatternNet/resolve/main/data/train-{i:05d}-of-00003.parquet",
            )
            for i in range(3)
        },
        citation="""@article{zhou2018patternnet,
            title={PatternNet: A benchmark dataset for performance evaluation of remote sensing image retrieval},
            author={Zhou, Weixun and Newsam, Shawn and Li, Congmin and Shao, Zhenfeng},
            journal={ISPRS Journal of Photogrammetry and Remote Sensing},
            volume={145},
            pages={197--209},
            year={2018},
            publisher={Elsevier}
        }""",
        license="Open Access for Research",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "PatternNet is a large-scale high-resolution remote sensing image dataset "
                "consisting of 30,400 RGB images across 38 classes (800 images per class, "
                "256x256 resolution). Partitioned into an 80/20 train/test split."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=CLASS_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()

        asset_keys = list(assets.keys())
        downloaded = bulk_download([assets[k] for k in asset_keys], dest_folder=download_dir)
        paths = dict(zip(asset_keys, downloaded))
        parquet_paths = [paths[f"train_{i}"] for i in range(3)]

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_paths": parquet_paths, "split": "train"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"data_paths": parquet_paths, "split": "test"}),
        ]

    def _generate_examples(self, data_paths, split: str = "train", **kwargs):
        """Generate examples from Apache Parquet files with deterministic 80/20 train/test partition."""
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        class_counters = [0] * len(CLASS_NAMES)
        yield_idx = 0

        for path in data_paths:
            table = pq.read_table(path)
            for row in table.to_pylist():
                label = int(row["label"])
                count_in_class = class_counters[label]
                class_counters[label] += 1

                is_train = count_in_class < 640
                if (split == "train" and not is_train) or (split == "test" and is_train):
                    continue

                img_data = row["image"]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    raw_bytes = img_data["bytes"]
                elif isinstance(img_data, bytes):
                    raw_bytes = img_data
                else:
                    raise ValueError(f"Unexpected image format: {type(img_data)}")

                with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
                    pil_rgb = pil_img.convert("RGB")
                    buf = io.BytesIO()
                    pil_rgb.save(buf, format="JPEG")
                    processed_bytes = buf.getvalue()

                yield (
                    yield_idx,
                    {
                        "image": processed_bytes,
                        "label": label,
                    },
                )
                yield_idx += 1
