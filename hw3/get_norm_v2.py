from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torch
from PIL import Image
import os
from tqdm import tqdm
# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    # transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomResizedCrop(128, scale=(0.4,1.0),ratio=(0.95,1.05)),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.4598, 0.3836, 0.3001],[0.3105, 0.2908, 0.2802]),
])

class FoodDataset(Dataset):
    def __init__(self,path,tfm=train_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

train_set = FoodDataset("train", tfm=train_tfm)
train_dl = DataLoader(train_set, 4, shuffle=True, num_workers=4)

def get_mean_std(dl):
    sum_, squared_sum, batches = 0,0,0
    for data, _ in tqdm(dl):
        sum_ += torch.mean(data, dim = ([0,2,3]))
        squared_sum += torch.mean(data**2, dim = ([0,2,3]))
        batches += 1
        
    mean = sum_/batches
    std = (squared_sum/batches - mean**2)**0.5
    return mean,std

mean, std = get_mean_std(train_dl)
print(mean)
print(std)
# tensor([0.4558, 0.4164, 0.3659])
# tensor([0.2995, 0.2909, 0.2952])
# tensor([0.4566, 0.4151, 0.3628])
# tensor([0.2995, 0.2898, 0.2934])