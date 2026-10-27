BreakHis
========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Classification-blue" alt="Task: Image Classification">
   <img src="https://img.shields.io/badge/Classes-8-green" alt="Classes: 8">
   <img src="https://img.shields.io/badge/Image%20Size-700x460-orange" alt="Image Size: 700x460">
   <img src="https://img.shields.io/badge/Domain-Histopathology-red" alt="Domain: Histopathology">
   </p>

Overview
--------

The Breast Cancer Histopathological Image Classification (BreakHis) dataset contains 7,909 microscopic images of breast tumour tissue, collected from 82 hematoxylin and eosin stained biopsy slides of 70 patients at the P&D Laboratory, Brazil. Each slide was imaged at four magnifications (40X, 100X, 200X and 400X), and each image is a 700x460 RGB PNG labelled with one of eight tumour types, four benign and four malignant.

BreakHis has no official split, and images of the same patient are near-duplicates of each other, so stable-datasets uses a fixed **patient-level** split (no patient appears in both):

- **Train**: 5,555 images from 48 patients
- **Test**: 2,354 images from 22 patients

Each tumour type has 29-35% of its images in the test split, except adenosis (14%), which comes from only two patients. Slides lettered AB, CD, ... of the same patient are grouped, and three patients have slides of two tumour types, which is why there are 70 patients for 82 slides.

.. figure:: teasers/breakhis_teaser.png
   :align: center
   :alt: BreakHis teaser samples

Data Structure
--------------

When accessing an example using ``ds[i]``, you receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - 700x460 RGB histopathology image
   * - ``label``
     - int
     - Tumour type, 0-7 (see Classes); ``label < 4`` is benign
   * - ``malignant``
     - int
     - ``0`` (benign) or ``1`` (malignant), for the binary task
   * - ``magnification``
     - int
     - ``0`` (40X), ``1`` (100X), ``2`` (200X) or ``3`` (400X)
   * - ``patient_id``
     - str
     - Patient identifier, e.g. ``"14-22549"``

Classes
-------

.. list-table::
   :header-rows: 1
   :widths: 10 30 20 40

   * - Label
     - Tumour type
     - Benign/malignant
     - Train / test images
   * - 0
     - adenosis
     - benign
     - 383 / 61
   * - 1
     - fibroadenoma
     - benign
     - 710 / 304
   * - 2
     - phyllodes_tumor
     - benign
     - 295 / 158
   * - 3
     - tubular_adenoma
     - benign
     - 387 / 182
   * - 4
     - ductal_carcinoma
     - malignant
     - 2,401 / 1,050
   * - 5
     - lobular_carcinoma
     - malignant
     - 429 / 197
   * - 6
     - mucinous_carcinoma
     - malignant
     - 554 / 238
   * - 7
     - papillary_carcinoma
     - malignant
     - 396 / 164

Usage Example
-------------

.. code-block:: python

    from stable_datasets.images import BreakHis

    # Downloads the 4.3 GB archive on first use
    ds = BreakHis(split="train")
    sample = ds[0]

    print(sample.keys())          # dict_keys(["image", "label", "malignant", "magnification", "patient_id"])
    print(sample["image"].size)   # (700, 460)
    print(sample["label"])        # tumour type, 0-7

    # Binary benign/malignant task at one magnification
    ds_200x = ds.filter(lambda ex: ex["magnification"] == 2)
    binary_label = ds_200x[0]["malignant"]

    # Load as PyTorch tensors
    ds_torch = ds.with_format("torch")
    print(ds_torch[0]["image"].shape)  # 3 x 460 x 700

Citation
--------

.. code-block:: bibtex

    @article{spanhol2016dataset,
        title={A Dataset for Breast Cancer Histopathological Image Classification},
        author={Spanhol, Fabio A and Oliveira, Luiz S and Petitjean, Caroline and Heutte, Laurent},
        journal={IEEE Transactions on Biomedical Engineering},
        volume={63},
        number={7},
        pages={1455--1462},
        year={2016},
        publisher={IEEE}
    }
