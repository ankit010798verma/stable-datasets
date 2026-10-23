import io

import pyarrow.parquet as pq
from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


class ChestXRay(BaseDatasetBuilder):
    """
    Chest X-Ray Images (Pneumonia) Dataset.

    The dataset contains 5,856 validated chest X-ray images (anterior-posterior)
    categorized into two classes: NORMAL and PNEUMONIA.
    Collected from pediatric patients aged 1-5 years old at Guangzhou Women and
    Children's Medical Center.

    Split sizes:
    - train: 5,216 images
    - validation: 16 images
    - test: 624 images
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://data.mendeley.com/datasets/rscbjbr9sj/3",
        assets={
            **{
                f"train_{i}": DownloadInfo(
                    url=f"https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia/resolve/main/data/train-{i:05d}-of-00007.parquet",
                )
                for i in range(7)
            },
            "validation": DownloadInfo(
                url="https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia/resolve/main/data/validation-00000-of-00001.parquet",
            ),
            "test": DownloadInfo(
                url="https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia/resolve/main/data/test-00000-of-00001.parquet",
            ),
        },
        citation="""@article{kermany2018identifying,
            title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
            author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and Valentim, Carolina CS and Liang, Huiying and Baxter, Sally L and McKeown, Alex and Yang, Ge and Wu, Xiaokang and Yan, Fangbing and others},
            journal={Cell},
            volume={172},
            number={5},
            pages={1122--1131},
            year={2018},
            publisher={Elsevier}
        }""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Chest X-Ray Images (Pneumonia) classification dataset consisting of 5,856 "
                "pediatric chest radiographs categorized into Normal and Pneumonia. "
                "All images are converted to 3-channel RGB."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=["NORMAL", "PNEUMONIA"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()

        asset_keys = list(assets.keys())
        downloaded = bulk_download([assets[k] for k in asset_keys], dest_folder=download_dir)
        paths = dict(zip(asset_keys, downloaded))

        train_paths = [paths[f"train_{i}"] for i in range(7)]
        val_path = paths["validation"]
        test_path = paths["test"]

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_paths": train_paths}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"data_paths": [val_path]}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"data_paths": [test_path]}),
        ]

    def _generate_examples(self, data_paths, **kwargs):
        """Generate examples from Apache Parquet files."""
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        idx = 0
        for path in data_paths:
            table = pq.read_table(path)
            for row in table.to_pylist():
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

                label = int(row["label"])
                yield idx, {
                    "image": processed_bytes,
                    "label": label,
                }
                idx += 1
