import torch
from einops import rearrange
import numpy as np

operation_seed_counter = 0

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, t, h, w = img.shape
    mask1 = torch.zeros(size=(n * t * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * t * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask3 = torch.zeros(size=(n * t * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor([
        [0, 1, 2], [0, 2, 1], 
        [1, 0, 3], [1, 3, 0], 
        [2, 0, 3], [2, 3, 0],
        [3, 2, 1], [3, 1, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * t * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * t * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    # [n * h // 2 * w // 2, ]
    rd_pair_idx = idx_pair[rd_idx]
    # [n * t * h // 2 * w // 2, 2]

    rd_pair_idx += torch.arange(start=0,
                                end=n * t * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    mask3[rd_pair_idx[:, 2]] = 1
    return mask1, mask2, mask3


def generate_subimages(img, mask):
    n, c, t, h, w = img.shape
    img = rearrange(img, 'b c s h w -> (b s) c h w')
    subimage = torch.zeros(n*t,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n*t, h // 2, w // 2, c).permute(0, 3, 1, 2)

    subimage = rearrange(subimage, '(n t) c h w -> n c t h w', n=n, t=t)
    return subimage


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    # cuda
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


class AugmentNoise_np(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * np.ones((shape[0], 1, 1))
            noise = np.random.normal(
                loc=0.0,
                scale=std,
                size=shape
            )
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.rand((shape[0], 1, 1)) * (max_std - min_std) + min_std
            noise = np.random.normal(
                loc=0.0,
                scale=std,
                size=shape
            )
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * np.ones((shape[0], 1, 1))
            noised = np.random.poisson(lam * x, size=shape) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.rand((shape[0], 1, 1)) * (max_lam - min_lam) + min_lam
            noised = np.random.poisson(lam * x, size=shape) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


if __name__ == "__main__":
    img = torch.randn((2, 10, 16, 64, 64))
    mask1, mask2 = generate_mask_pair(img)

    noisy_sub1 = generate_subimages(img, mask1)
    noisy_sub2 = generate_subimages(img, mask2)
    print(noisy_sub1.shape)
    print(noisy_sub2.shape)


