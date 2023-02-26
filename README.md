# SRDTrans

<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

👀[**Demos**](#-demos-videos) **|** 🚩[**Updates**](#-updates) **|** ⚡[**Usage**](#-quick-inference) **|** 🏰[**Model Zoo**](docs/model_zoo.md) **|** 🔧[Install](#-dependencies-and-installation)  **|** 💻[Train](docs/Training.md) **|** ❓[FAQ](docs/FAQ.md) **|** 🎨[Contribution](docs/CONTRIBUTING.md)

</div>

## 🔧 Dependencies and Installation

- Python>=3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

  ### Installation

  1. Clone repo

      ```bash
      git clone https://github.com/Huxiaowan/SRDTrans.git
      cd SRDTrans
      ```

  1. Install dependent packages

      ```bash
      $ conda create -n srdtrans python=3.6
      $ conda activate srdtrans
      $ pip install -r requirements.txt
      ```


## 💻 Train 
  Download the demo data(.tif file) [DataForPytorch] and put it into SRDTrans/datasets/

  ```bash
  	$ python -u train.py \
      --datasets_folder 0.3hz \
      --datasets_path [path to noisy tif] \
      --n_epochs 30 \
      --GPU 1 \
      --train_datasets_size 3000 \
      --patch_x 128 \
      --patch_t 128 \
      --clean_img_path [path to clean tif]

  ```


## 🏰 Model Zoo
| Models                            | Modality  | Scale | Description                                  |
| --------------------------------- |:--------- | :---- | :------------------------------------------- |
| SRDTrans_Calcium                  | Calcium   |       |                                              |
| SRDTrans_STORM                    | STORM     |       |                                              |


## ⚡ Quick Inference
```bash
  $ python test.py \
    --denoise_model [mdoel_name] \
    --patch_x 128 \
    --patch_t 128 \
    --GPU 5 \
    --ckp_idx 9 \
    --datasets_folder [folder name] \
    --test_datasize 1000 \
    --datasets_path [path to noisy tif] \
    --clean_path [path to clean tif]
```

---

## BibTeX

    @InProceedings{wang2021realesrgan,
        author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
        date      = {2021}
    }
