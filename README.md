# WDSS

## Installation
To set up the environment, please follow these steps:

1. Create and activate a conda environment :
    ```shell
    conda create -n wdss_env python=3.7
    conda activate wdss_env
    ```

2. Install the required packages:
    ```shell
    pip install -r requirements.txt
    ```

## Datasets
You can download the pre-processed datasets from the following Baidu Netdisk link:

Vaihingen: [Baidu Netdisk Link](your_baidu_link)

Loveda: [Baidu Netdisk Link](your_baidu_link)

Potsdam: [Baidu Netdisk Link](your_baidu_link)


After downloading and extracting, place the datasets in the `./Datasets` directory.

## Project Structure

The folder structure for the `WDSS` directory is as follows:

```plaintext
WDSS/
├── Datasets/
│   ├── Loveda/
│   │   ├── 3SGB/
│   │   ├── 3SGO/
│   │   ├── img/
│   │   ├── img_0.125/
│   │   ├── img_0.25/
│   │   ├── label/
│   │   ├── label_0.125/
│   │   ├── label_0.25/
│   │   ├── SGB/
│   │   └── SGO/
│   ├── Potsdam/
│   └── Vaihingen/
├── Result/
│   ├── loveda/
│   │   ├── 0.125/
│   │   │   ├── 1/
│   │   │   │   ├── img_PL/
│   │   │   │   └── lable_PL/
│   │   │   ├── 2/
│   │   ├── 0.25/
│   │   │   ├── 1/
│   │   │   │   ├── img_PL/
│   │   │   │   └── lable_PL/
│   │   │   ├── 2/
│   ├── potsdam/
│   └── vaihingen/
├── pretrain/
│   ├── sam_vit_h_4b8939.pth
│   ├── swinv2_base_patch4_window16_256.pth
├── model/
│   ├── FTUNetFormer.py
│   ├── FTUNetFormer_1.py
│   └── swintransformerv2.py
├── tools/
│   ├── image_cropping.py
│   └── SAM_wavelet.py
├── mask.py
├── requirements.txt
├── train_1.py
├── train_2.py
├── utils_1.py
├── utils_2.py
└── utils_mask.py
```




## Usage

### Two-Stage Self-Training 
You can directly download our pre-processed datasets and use them for the second stage of training without generating SGO and SGB or performing the first stage of training to prepare pseudo-labels.

#### Second Stage Training
1. Modify the `train_2.py` and `utils_2.py` files according to your paths and datasets, and load 3SGO and 3SGB.
2. Navigate to the project directory and activate the environment:
    ```shell
    conda activate wdss_env
    ```
3. Run the following command to start training:
    ```shell
    python train_2.py
    ```

### Complete Two-Stage Process
If you want to use your own datasets and perform the complete two-stage process, you can use `image_cropping.py` and `SAM_wavelet.py` provided in the `tools` folder for image cropping and wavelet-driven SAM-generated object reconstruction.


#### First Stage Training
1. Perform the first stage of training by loading SGO and SGB, and modifying the `train_1.py` and `utils_1.py` files.
2. Navigate to the project directory and activate the environment:
    ```shell
    conda activate wdss_env
    ```
3. Run the following command to start training:
    ```shell
    python train_1.py
    ```

4. Generate pseudo-labels using `mask.py` and `utils_mask.py`:
    ```shell
    python mask.py
    ```

