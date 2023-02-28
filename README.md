# Spatial redundancy transformer for self-supervised fluorescence image denoising

<p align="center">
  <img src="assets/SRDTrans.gif" width='600'>
</p>

## <div align="center"><b><a href="README.md">SRDTrans</a></div>

<div align="center">

✨[**Method**][def] **|** 🚩[**Paper**](#-updates) **|** 🔧[**Install**](#-dependencies-and-installation)  **|** 🎨[**Data preparation**](docs/CONTRIBUTING.md)**|** 🏰[**Model Zoo**](docs/model_zoo.md)  **|** 💻[**Train**](docs/Training.md) **|** ⚡[**Usage**](#-quick-inference)**|** &#x1F308;[**Demo**]()**|** ❓[FAQ](docs/FAQ.md) 

</div>


---


## ✨ Method

  
  <p align="center">
  <img src="assets/self_super.png" width='800'>
  </p>

  + A spatial redundancy denoising transformer (SRDTrans) network is proposed to remove detection noise from fluorescence time-lapse images in a self-supervised manner. We designed a sampling strategy to extract training pairs based on spatial redundancy and orthogonal correlation, which eliminates the dependence on high imaging speed. 
  
  + To break the performance bottleneck of convolutional neural networks (CNNs) on fluorescence image denoising, we designed a 3D Transformer network to improve the model capability to capture long-range spatiotemporal dependencies and high-resolution features. SRDTrans can overcome the inherent spectral bias of CNNs and restore realistic edges and textures in denoised images instead of bringing over-smooth structures and distorted pixels. 
  
  + We demonstrate the state-of-the-art denoising performance of SRDTrans on simulated and experimental data of two-photon calcium imaging and single-molecule localization microscopy. SRDTrans does not contain any assumptions about the imaging process and the sample, thus can be easily extended to a wide range of applications.




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
  




[def]: #-demos-videos