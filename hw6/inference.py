# Generate 1000 images and make a grid to save them.
import os
import torch
import torchvision
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

loader = ModelLoader(
    base_dir = '.',   # path to where you invoked the command line tool
    name = 'stylegan2', # the project name, defaults to 'default'
    # load_from = 40
)


n_output = 1000
noise   = torch.randn(n_output, 512) # noise
styles  = loader.noise_to_styles(noise, trunc_psi = 0.75)  # pass through mapping network

eval_batch_size = 200
def inference(self, num=1000, n_iter=5, output_path='./submission2'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with torch.no_grad():
            for i in range(n_iter):
                all_images = loader.styles_to_images(styles[i*eval_batch_size:(i+1)*eval_batch_size])
                for j in range(eval_batch_size):
                    torchvision.utils.save_image(all_images[j], f'{output_path}/{i * eval_batch_size + j + 1}.jpg')
inference(loader, num=1000, n_iter=5, output_path='./submission_stylegan2')
