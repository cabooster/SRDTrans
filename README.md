# Spatial redundancy transformer for self-supervised fluorescence image denoising

<p align="center">
  <img src="assets/SRDTrans.gif" width='600'>
</p> 

## <div align="center"><b><a href="README.md">SRDTrans</a></b></div>

<div align="center">

✨ [**Method**][def] **|** 🚩 [**Paper**](#-updates) **|** 🔧 [**Install**](#-dependencies-and-installation)  **|** 🎨 [**Data preparation**](docs/CONTRIBUTING.md)**|** 🏰 [**Model Zoo**](docs/model_zoo.md)  **|** 💻 [**Train**](docs/Training.md) **|** ⚡ [**Usage**](#-quick-inference)**|** &#x1F308; [**Demo**]()

</div>

---


## ✨ Method

<p align="center">
<img src="assets/self_super.png" width='800'>
</p>

- The spatial redundancy denoising transformer (SRDTrans) network is proposed to remove detection noise from fluorescence time-lapse images in a self-supervised manner. We designed a sampling strategy to extract training pairs based on spatial redundancy and orthogonal correlation. This scheme has no dependence on the similarity between two adjacent frames, so our SRDTrans is applicable to very fast dynamics and extremely low imaging speed, which is totally complementary to our previously proposed [DeepCAD](https://www.nature.com/articles/s41592-021-01225-0) and [DeepCAD-RT](https://www.nature.com/articles/s41587-022-01450-8). We demonstrate the state-of-the-art denoising performance of SRDTrans on simulated and experimental data of two-photon calcium imaging and single-molecule localization microscopy. SRDTrans does not contain any assumptions about the imaging process and the sample, thus can be easily extended to a wide range of applications.



## 🚩 Paper

This repository is for RCAN introduced in the following paper

"Spatial redundancy transformer for self-supervised fluorescence image denoising" [[arXiv]](ht) 




## 🔧 Install

### Dependencies 
  - Python >= 3.6 
  - PyTorch >= 1.7
    
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
|Calcium_30Hz                  | 5.44 G   |  [Drive](https://drive.google.com/drive/folders/1NpwKws3Wt4gQzai4b_HOshR9cBkxPb9v?usp=share_link)     |         
|Calcium_0.3Hz                  | 927.40 M   |  [Drive](https://drive.google.com/drive/folders/1rB4s46cw1yyVbyNabTvqYKt7OerPkkG4?usp=share_link)     |                                       |
|STORM                    |   2.00 G     |    [Drive](https://drive.google.com/drive/folders/1QbQkJl85v9i5YFdOHJRqxfrDfSx-C0lK?usp=share_link)    |                                              |

Download the demo data(.tif file) and put it into SRDTrans/datasets/.

## 💻 Train 

1. Data preparation 
  
    Download the demo data(.tif file) and put the noisy data and the clean data into datasets/noisy/ and  datasets/clean/, respectively.

2. Train

You can use scripts in file 'train_SRDTrans.sh' to train models for our paper.
  ```bash
    # Simulated STORM & Simulated Calcium imaging data at ~30hz
    python -u train.py --datasets_folder noisy --datasets_path datasets/ --n_epochs 30 --GPU 0 --train_datasets_size 6000  --patch_x 128 --patch_t 128 --clean_img_path datasets/clean/clean.tif
    # Simulated Calcium imaging data at ~0.3hz
    python -u train.py --datasets_folder noisy --datasets_path datasets/ --n_epochs 30 --GPU 0 --train_datasets_size 3000  --patch_x 128 --patch_t 128 --clean_img_path datasets/clean/clean.tif
  ```


## 🏰 Model Zoo
| Models                            | Modality  |Download | Description                                  |
| --------------------------------- |:--------- | :---- | :------------------------------------------- |
| SRDTrans_Calcium                  | Calcium   |  [Drive](https://drive.google.com/drive/folders/19y5gmqoUwEw7rMFhVLHW4ZAe1X6HwzV4?usp=share_link)     | The pretrained model on calcium imaging data.                                            |
| SRDTrans_STORM                    | STORM     |    [Drive](https://drive.google.com/drive/folders/188rt3adrnVbF7zIxMgg2NRQJKl-e1_Fl?usp=share_link)    | The pretrained model on STORM  data.                                           | 


## ⚡ Quick Inference
1. Pretrained Model

    Download the pretrained model of STORM and put it into pth/storm.

    Download the pretrained model of Calcium imaging data at ~30hz and put it into pth/cad_30hz.

    Download the pretrained model of Calcium imaging data at ~0.3hz and put it into pth/cad_03hz.

2. Test

  You can use scripts in file 'test_SRDTrans.sh' to train models for our paper.
  ```bash
    # Simulated STORM 
    python -u test.py --denoise_model storm --patch_x 128 --patch_t 128 --GPU 0 --ckp_idx 9 --datasets_folder noisy --test_datasize 1000 --datasets_path datasets/ --clean_path datasets/clean/clean.tif
    # Simulated Calcium imaging data at ~30hz
    python -u test.py --denoise_model cad_30hz --patch_x 128 --patch_t 128 --GPU 0 --ckp_idx 9 --datasets_folder noisy --test_datasize 1000 --datasets_path datasets/ --clean_path datasets/clean/clean.tif
    # Simulated Calcium imaging dataat ~0.3hz
    python -u test.py --denoise_model cad_03hz --patch_x 128 --patch_t 128 --GPU 0c --ckp_idx 9 --datasets_folder noisy --test_datasize 1000 --datasets_path datasets/ --clean_path datasets/clean/clean.tif
  ```

---
## &#x1F308; Demo

  ### 1. STORM Denoising
  <p align="center">
  <img src="assets/storm_vis.png" width='800'>
  </p>

  ### 2. Localization and reconstruction of STORM
  <p align="center">
  <img src="assets/storm_rec.png" width='800'>
  </p>

  ### 3. Calcium Denoising at ~30Hz
  <p align="center">
  <img src="assets/cad_30hz.png" width='800'>
  </p>

  ### 4. Calcium Denoising at ~0.3Hz
  <p align="center">
  <img src="assets/cad_0.3hz.png" width='800'>
  </p>

  ### 4. Videos
  1. Three-dimensional visualization of large neuronal populations in a 510 × 510 × 548 μm volume (100 planes, 0.3-Hz volume rate). Left, low-SNR raw volume. Right, the same volume denoised with SRDTrans.
  <p align="center">
    <img src="assets/Supplementary Video_5.gif" width='500'>
    </p>
  2. Comparison of denoising performance of DeepCAD, CNN, and SRDTrans at ~0.3Hz calcium imaging data.
  <p align="center">
    <img src="assets/Supplementary Video_4.gif" width='500'>
    </p>
  2. Validation experiments on synthetic MNIST datasets of different moving speed [0.5,0.5,10] (pixel/s). Faster movement of digits in MNIST means a lager discrepancy gap between adjacent frames. 

  <p align="center">
    <img src="assets/Supplementary Video_3.gif" width='500'>
    </p>


[def]: #-demos-videos