## Off-the-shelf Vision Models Benefit Image Manipulation Localization


This repository represents the official implementation of the paper titled "Off-the-shelf Vision Models Benefit Image Manipulation Localization".

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2604.09096)  

[Zhengxuan Zhang](https://github.com/Johnson-u) , [Keji Song](https://github.com/Stech0109) , [Junmin Hu](https://github.com/JunminHuOUC) , [Ao Luo]() , [Yuezun Li](https://yuezunli.github.io)<sup>#</sup>, [<img src="./fig/logo.png" alt="Logo" width="28" height="21" />](https://yuezunli.github.io/ligroup/) [Vas Group](https://yuezunli.github.io/ligroup/)

<sup>#</sup>Corresponding author.

<p align="center">
  <img src="./fig/overview.png" width="48%" style="margin-right:1%;" />
  <img src="./fig/ria.jpg" width="48%" />
</p>

This code corresponds to the **ReVi** implementation based on the **[TinySAM](https://github.com/xinghaochen/TinySAM)** framework.

## 🛠️ Environment & Setup

The inference code was tested on:

Ubuntu 22.04 LTS, Python 3.10.12,  CUDA 12.8, GeForce RTX 3090ti



You can create a virtual environment and install the required dependency libraries in the following:

```python
pip install virtualenv
mkdir venv && cd venv
virtualenv ReVi
source ReVi/bin/activate
pip install -r requirements.txt
```



## 📦 Prepare for your dataset


 

## 🏋️ Training




## 🏃 Visualization and Testing

1. The testing configuration for **ReVi** is located in the `./infer_config.yaml` file. You can modify the relevant information in the file, especially the `checkpoint`, `input_folder`, `gt_folder`, and `output_folder`. After making the changes, you can run the testing code using the following command.
    ```python
   python infer.py
   ```
   Alternatively, you can also create a custom configuration file and test with the following command.
    ```python
   python infer.py --config "your config file"
   ```
   During the testing process, the visual localization results will be output to `output_folder`, and the **AUC** and **F1** scores of the test results will be reported.
   

2. We provide the [weights](https://pan.baidu.com/s/1oUzrqwsEFksXcp_UW59slg?pwd=82dn) of the **ReVi** version under the **Pre-trained protocol** (trained on the PSCC dataset) based on the TinySAM model. You can download it and place the files in `checkpoint`.


3. **Note: The filenames of your tampered image files and ground truth files must be consistent (extensions can differ); otherwise, an error will occur when calculating metrics. Also, please ensure that the two files have the same size.**


[//]: # (**Note: The model we are testing here is the model of the previous work that was trained using our method.**)



## ⏳ Todo List
To facilitate comparisons with other research efforts, we are working on integrating eight baseline methods and the four models from this paper into a **unified framework**. The specific codebase and usage details will be made public in the coming months.

## 🎓 Citation

Thanks for your attention, please cite our paper:

```bibtex
@misc{zhang2026offtheshelfvisionmodelsbenefit,
      title={Off-the-shelf Vision Models Benefit Image Manipulation Localization}, 
      author={Zhengxuan Zhang and Keji Song and Junmin Hu and Ao Luo and Yuezun Li},
      year={2026},
      eprint={2604.09096},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.09096}, 
}
```
