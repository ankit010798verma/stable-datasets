import re

import pyarrow.parquet as pq

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


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

# Each class has images numbered 001-800; this builder puts 001-640 in train and 641-800 in test.
TRAIN_IMAGES_PER_CLASS = 640

_FILE_NUMBER_RE = re.compile(r"(\d+)\.jpg$")


class PatternNet(BaseDatasetBuilder):
    """
    PatternNet Remote Sensing Image Classification Dataset.

    PatternNet is a large-scale high-resolution remote sensing dataset collected for
    performance evaluation of remote sensing image retrieval and scene classification.
    Images are collected from Google Earth or the Google Maps API.

    The dataset contains 38 classes with 800 images per class (30,400 images total).
    Each image is 256x256 pixels in RGB format.

    PatternNet has no official split. This builder splits each class by the
    number in its file names (e.g. ``airplane001.jpg``):

    - train: images 001-640 of each class, 24,320 images
    - test: images 641-800 of each class, 6,080 images
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
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "PatternNet is a large-scale high-resolution remote sensing image dataset "
                "consisting of 30,400 RGB images across 38 classes (800 images per class, "
                "256x256 resolution), split 80/20 per class by image number into train and test."
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
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        asset_keys = list(assets.keys())
        downloaded = bulk_download([assets[k] for k in asset_keys], dest_folder=self._raw_download_dir)
        paths = dict(zip(asset_keys, downloaded))
        parquet_paths = [paths[f"train_{i}"] for i in range(3)]

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_paths": parquet_paths, "split": "train"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"data_paths": parquet_paths, "split": "test"}),
        ]

    def _generate_examples(self, data_paths, split):
        """Yield the source JPEG bytes unchanged, keeping images 001-640 of each class for train and the rest for test."""
        yield_idx = 0
        for path in data_paths:
            for row in pq.read_table(path).to_pylist():
                file_number = int(_FILE_NUMBER_RE.search(row["image"]["path"]).group(1))
                if (file_number <= TRAIN_IMAGES_PER_CLASS) != (split == "train"):
                    continue
                yield yield_idx, {"image": row["image"]["bytes"], "label": int(row["label"])}
                yield_idx += 1
