from torch.utils import data
from PIL import Image
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for _, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
    return images[:len(images)]


class InputData(data.Dataset):
    def __init__(self, root, max_epoch=None, crop_size=(512, 512)):
        self.root = root
        self.crop_size = crop_size
        self.img_ids = sorted(make_dataset(self.root+'/image'))
        if not max_epoch==None:
            self.img_ids = self.img_ids * int(max_epoch)

        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "image/%s" % name)
            label_file = osp.join(self.root, "label/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('1')

        name = datafiles["name"]

        transform_img_list = []
        transform_label_list =[]
        transform_img_list.append(transforms.Resize(self.crop_size, Image.BICUBIC))

        transform_img_list += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]

        transform_img = transforms.Compose(transform_img_list)
        image = transform_img(image)

        label = label.resize(self.crop_size, Image.NEAREST)
        label = np.asarray(label, np.float32)

        return image, label, name


def transform_style_img(image_pth):
    img = Image.open(image_pth).convert('RGB')
    transform_img_list = []
    transform_img_list += [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    transform_img = transforms.Compose(transform_img_list)
    image = transform_img(img)
    image = image.unsqueeze(0)
    image = image.cuda()
    return image

if __name__ =='__main__':
    path = '/media/lmy/lmy-2/building/tianjin/'
    loader = InputData(path)




