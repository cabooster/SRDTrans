# Spatial redundancy transformer for self-supervised fluorescence image denoising

<p align="center">
  <img src="assets/SRDTrans.gif" height=180>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

✨[**Method**][def] **|** 🚩[**Paper**](#-updates) **|** 🔧[**Install**](#-dependencies-and-installation)  **|** 🎨[**Data preparation**](docs/CONTRIBUTING.md)**|** 🏰[**Model Zoo**](docs/model_zoo.md)  **|** 💻[**Train**](docs/Training.md) **|** ⚡[**Usage**](#-quick-inference)**|** &#x1F308;[**Demo**]()**|** ❓[FAQ](docs/FAQ.md) 

</div>

## Method

  
  <p align="center">
  <img src="assets/self_super.png" height=180>
  </p>

  




## 🔧 Install

  ### Dependencies 

      Python >= 3.6 
      PyTorch >= 1.7
    

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

## 🎨 Data preparation

| Data                            | Size  |Download |                                  |
| --------------------------------- |:--------- | :---- | :------------------------------------------- |
|Calcium_30Hz                  | 5.44 G   |  [Drive]()     |         
|Calcium_0.3Hz                  | 927.40 M   |  [Drive]()     |                                       |
|STORM                    |   2.00 G     |    [Drive]()    |                                              |

Download the demo data(.tif file) and put it into SRDTrans/datasets/.

## 💻 Train 

  ```bash
  	$ python -u train.py \
      --datasets_folder [folder name] \
      --datasets_path [data_path] \
      --n_epochs 30 \
      --train_datasets_size 6000 \
      --patch_x 128 \
      --patch_t 128 \
      --clean_img_path [gt_path]

  ```


## 🏰 Model Zoo
| Models                            | Modality  |Download | Description                                  |
| --------------------------------- |:--------- | :---- | :------------------------------------------- |
| SRDTrans_Calcium                  | Calcium   |  [Drive]()     |                                              |
| SRDTrans_STORM                    | STORM     |    [Drive]()    |                                              |



## ⚡ Quick Inference
```bash
  $ python test.py \
    --denoise_model [model_name] \
    --patch_x 128 \
    --patch_t 128 \
    --ckp_idx [the optimal epoch] \
    --datasets_folder [folder name] \
    --test_datasize 1000 \
    --datasets_path [data_path] \
    --clean_path [gt_path to clean tif]
```

---
## ⚡ Demo

  ### 1. STORM Denoising
  <p align="center">
  <img src="assets/storm_vis.png" height=170>
  </p>

  ### 2. Localization and reconstruction of STORM
  <p align="center">
  <img src="assets/storm_rec.png" height=160>
  </p>

  ### 3. Calcium Denoising at ~30Hz
  <p align="center">
  <img src="assets/cad_30hz.png" height=120>
  </p>

  ### 4. Calcium Denoising at ~0.3Hz
  <p align="center">
  <img src="assets/cad_0.3hz.png" height=300>
  </p>

  ### 4. Videos
  





## BibTeX

    @InProceedings{wang2021realesrgan,
        author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
        date      = {2021}
    }


[def]: #-demos-videos