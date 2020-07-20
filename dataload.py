import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import transforms

class Image_Transform():
    def __init__(self, resize=384, mean=(0.485, 0.224, 0.406), std=(0.229, 0.224, 0.225), train=True):
        self.data_transform = {
            'img': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'rd': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
            ])
        }
        if train == False:
            self.data_transform['img'] = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
            ])
    
    def __call__(self, img, img_or_rd='img'):
        return self.data_transform[img_or_rd](img)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, test=False):
        self.imgs = glob.glob('./redweb/imgs/*.jpg')
        self.rds = glob.glob('./redweb/rds/*.png')
        if test:
            self.imgs = glob.glob("./test_imgs/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img, img_or_rd='img')

        rd_path = self.rds[idx]
        rd = Image.open(rd_path)
        rd_transformed = self.transform(rd, img_or_rd='rd')
        
        return img_transformed, rd_transformed

