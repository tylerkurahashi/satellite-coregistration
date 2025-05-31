# SpaceNet 9
Optical to SAR image registration.

## Prepare data

1. Download data from `https://discussions.topcoder.com/discussion/35945/download-links`
2. Unzip `train` and `publictest` and put `.tif` and `.csv` files in `/00_data/raw/train` and `/00_data/raw/test`


## Data Description
For training, there are three regions. Each region includes a SAR image scene (GeoTIFF), optical image scene (GeoTIFF), and tiepoints (CSV)

```bash
.
├── publictest
│   ├── 001
│   │   ├── 0_optical_publictest.tif
│   │   └── 1_sar_publictest.tif
│   └── 002
│       ├── 03_optical_publictest.tif
│       └── 03_sar_publictest.tif
└── train
    ├── 001
    │   ├── 0_optical.tif
    │   ├── 1_sar.tif
    │   └── 2_tiepoints.csv
    ├── 002
    │   ├── 0_optical.tif
    │   ├── 1_sar.tif
    │   └── 2_tiepoints.csv
    └── 003
        ├── 0_optical.tif
        ├── 1_sar.tif
        └── 2_tiepoints.csv
```

#### Optical imagery
- 3 channels with order (red, green, blue)
- high resolution imagery with 0.3 - 0.5 meter cell size
- provider: Maxar

#### SAR imagery
- single channel
- high resolution iamgery with 0.3 - 0.5 meter cell size
- provider: Umbra

#### Tiepoints
Each row represents a tiepoint between the images indicated by the row or column in the source image.

first three rows of `02_tiepoints_train_01.csv` 
| sar_row | sar_col | optical_row | optical_col |
|---------|---------|-------------|-------------|
| 189    | 4696    | 430        | 6723        |
| 209    | 408    | 499        | 794        |
| 238    | 296    | 537        | 635        |

In this example, the pixel located at sar_row=189 and sar_col=4696 in the SAR image (02_sar_train_02.tif) corresponds to the pixel located at optical_row=430 and optical_col=6723 in the optical image (02_optical_train_02.tif).

## Baseline Algorithm
- [SpaceNet 9 Basline Algorithm](./src/baseline/README.md)

## Submission
To reproduce the submission, we need:
- An `inference.py` script should output a `prediction_offset.tif` geotiff image. A valid `prediction_offset.tif` has the following characteristics:
    - two channels. Indicates how many pixels in the x and y direction to shift from a pixel in the optical image in order to find the matching SAR pixel. Pixel values in the first channel of the offset image should indicate the number of pixels in the x-direction to shift in order to find the corresponding pixel in the SAR image. Pixel values in second channel of the offset image indicate the y-direction shift.
    - The offset image should have the same pixel size as the optical image
    - The offset image should have the same spatial extent and number of rows and columns as the input optical image.
    - The offset image should have the same coordinate reference system as the the input optical image.

## Evaluation
- Evaluation metric is average distance between predicted offset and reference SAR tiepoint.

## References and Resources
- Hänsch, R., Arndt, J., Dias, P., Potnis, A., Lunga, D., Petrie, D., & Bacastow, T. (2024). *Introducing SpaceNet 9 - Cross-Modal Satellite Imagery Registration for Natural Disaster Responses*. In **IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium** (pp. 234-238). IEEE. [https://doi.org/10.1109/IGARSS53475.2024.10640611](https://doi.org/10.1109/IGARSS53475.2024.10640611)