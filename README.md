# WDSS

## Installation
To set up the environment, please follow these steps:

1. Create and activate a conda environment with Python 3.7:
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

[Baidu Netdisk Link](your_baidu_link)

After downloading and extracting, place the datasets in the `WDSS/Datasets` directory.

## Project Structure




## Usage

### Two-Stage Self-Training Framework
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
If you want to use your own datasets and perform the complete two-stage process, you can use the tools provided in the `tools` folder to perform image cropping and SAM object reconstruction.

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

## Contributing
We welcome contributions to the WDSS project! Here are some ways you can contribute:
- Report bugs and issues
- Submit feature requests
- Write or improve documentation
- Submit pull requests for code improvements or new features

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
We would like to thank all the contributors and the open-source community for their support.

## Contact
If you have any questions or need further assistance, please feel free to contact us at dehaozhou@example.com.

---

Thank you for using WDSS! We hope it helps you in your research and projects.

