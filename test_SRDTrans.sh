## test

## 1. Pretrained Model
# Download the pretrained model of STORM and put it into pth/storm.
# Download the pretrained model of Calcium imaging data at ~30hz and put it into pth/cad_30hz.
# Download the pretrained model of Calcium imaging data at ~0.3hz and put it into pth/cad_03hz.

## 2. Test
# Simulated STORM 
##################################################################################################################################
python -u test.py --denoise_model storm --patch_x 128 --patch_t 128 --GPU 0 --ckp_idx 9 --datasets_folder noisy --test_datasize 1000 --datasets_path datasets/ --clean_path datasets/clean/clean.tif

# Simulated Calcium imaging data at ~30hz
##################################################################################################################################
python -u test.py --denoise_model cad_30hz --patch_x 128 --patch_t 128 --GPU 0 --ckp_idx 9 --datasets_folder noisy --test_datasize 1000 --datasets_path datasets/ --clean_path datasets/clean/clean.tif


# Simulated Calcium imaging dataat ~0.3hz
##################################################################################################################################
python -u test.py --denoise_model cad_03hz --patch_x 128 --patch_t 128 --GPU 0c --ckp_idx 9 --datasets_folder noisy --test_datasize 1000 --datasets_path datasets/ --clean_path datasets/clean/clean.tif
