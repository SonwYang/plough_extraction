from torch.utils.data import DataLoader, Dataset
import torch
import cv2 as cv
import numpy as np
import os
import glob
import random
from skimage.morphology import dilation, square
from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
import torchvision

def img_to_tensor(img):
    tensor = torch.from_numpy(img.transpose((2, 0, 1)))
    return tensor


def to_monochrome(x):
    # x_ = x.convert('L')
    x_ = np.array(x).astype(np.float32)  # convert image to monochrome
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_


class TestDataset(Dataset):
    def __init__(self, root, mode='train', is_ndvi=False):
        self.root = root
        self.mode = mode
        self.mean_bgr = [104.00699, 116.66877, 122.67892]
        self.is_ndvi = is_ndvi
        self.imgList = sorted(img for img in os.listdir(self.root))
        self.imgTransforms = Compose([img_to_tensor])
        self.maskTransforms = Compose([
            torchvision.transforms.Lambda(to_monochrome),
            torchvision.transforms.Lambda(to_tensor),
        ])

    def __getitem__(self, idx):
        imgPath = os.path.join(self.root, self.imgList[idx])
        img = cv.imread(imgPath, cv.IMREAD_COLOR)
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img /= 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        imgName = os.path.split(imgPath)[-1].split('.')[0]

        if self.mode == 'test':
            batch_data = {'img': img, 'file_name': imgName}
            return batch_data

    def __len__(self):
        return len(self.imgList)


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    return dst


class RandomFlip:
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, img, mask=None, wm=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = np.flip(img, d)
            if mask is not None:
                mask = np.flip(mask, d)
                wm = np.flip(wm, d)
        return img, mask, wm


class RandomRotate90:
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, img, mask=None, wm=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
                wm = np.rot90(wm, factor)

        return img.copy(), mask.copy(), wm.copy()


class Rescale(object):
    def __init__(self, output_size, prob=0.9):
        self.prob = prob
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, image, label, wm=None):
        if random.random() < self.prob:
            raw_h, raw_w = image.shape[:2]

            img = cv.resize(image, (self.output_size, self.output_size))
            lbl = cv.resize(label, (self.output_size, self.output_size))
            wm = cv.resize(wm, (self.output_size, self.output_size))


            h, w = img.shape[:2]

            if h > raw_w:
                i = random.randint(0, h - raw_h)
                j = random.randint(0, w - raw_h)
                img = img[i:i + raw_h, j:j + raw_h]
                lbl = lbl[i:i + raw_h, j:j + raw_h]
                wm = wm[i:i + raw_h, j:j + raw_h]

            else:
                res_h = raw_w - h
                img = cv.copyMakeBorder(img, res_h, 0, res_h, 0, borderType=cv.BORDER_REFLECT)
                lbl = cv.copyMakeBorder(lbl, res_h, 0, res_h, 0, borderType=cv.BORDER_REFLECT)
                wm = cv.copyMakeBorder(wm, res_h, 0, res_h, 0, borderType=cv.BORDER_REFLECT)

            return img, lbl, wm
        else:
            return image, label, wm


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None, wm=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            height, width = img.shape[0:2]
            mat = cv.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv.warpAffine(img, mat, (height, width),
                                 flags=cv.INTER_LINEAR,
                                 borderMode=cv.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv.warpAffine(mask, mat, (height, width),
                                      flags=cv.INTER_LINEAR,
                                      borderMode=cv.BORDER_REFLECT_101)
                wm = cv.warpAffine(wm, mat, (height, width),
                                     flags=cv.INTER_LINEAR,
                                     borderMode=cv.BORDER_REFLECT_101)

        return img, mask, wm


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, wm=None):
        for t in self.transforms:
            x, mask, wm = t(x, mask, wm)
        return x, mask, wm


class BIPEDDataset(Dataset):
    def __init__(self, img_root, mode='train', crop_size=None):
        scaleList = [int(crop_size * 0.75),
                     int(crop_size * 0.875),
                     crop_size,
                     int(crop_size * 1.125),
                     int(crop_size * 1.25)]
        self.img_root = img_root
        self.mode = mode
        self.imgList = os.listdir(img_root)
        self.crop_size = crop_size
        self.transforms = DualCompose([
                Rotate(),
                RandomFlip(),
                RandomRotate90(),
                Rescale(scaleList[random.randint(0, len(scaleList) - 1)])
            ])

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.img_root, self.imgList[idx])
        assert os.path.exists(imgPath), 'please check if the image path exists'
        labelRoot = self.img_root.replace('images', 'labels')
        file_name = self.imgList[idx].split('.')[0]
        labelPath = glob.glob(f'{labelRoot}/{file_name}*')[0]
        suffix = self.imgList[idx].split('.')[-1]
        #####load data
        if suffix == 'npy':
            image = np.load(imgPath)
        else:
            image = cv.imread(imgPath, cv.IMREAD_COLOR)
            # image = cv.bilateralFilter(image, 9, 75, 75)
        label = cv.imread(labelPath, cv.IMREAD_GRAYSCALE)
        edge = np.where(label == 2, 1, 0)
        ploy = np.where(label == 1, 1, 0)
        wm1 = distranfwm(edge, beta=5)
        wm2 = distranfwm(ploy, beta=5)
        wm = wm1 + wm2
        image_shape = [image.shape[0], image.shape[1]]
        image, label, skeleton0, wm = self.transform(img=image, gt=label, wm=wm)
        return dict(images=image, labels=label, weight_mapping=wm, edges=skeleton0,
                    file_name=file_name, image_shape=image_shape)

    def transform(self, img, gt, wm):
        if self.crop_size:
            h, w = img.shape[:2]
            assert (self.crop_size < h and self.crop_size < w)
            i = random.randint(2, h - self.crop_size-2)
            j = random.randint(2, w - self.crop_size-2)
            img = img[i:i + self.crop_size, j:j + self.crop_size]
            gt = gt[i:i + self.crop_size, j:j + self.crop_size]
            wm = wm[i:i + self.crop_size, j:j + self.crop_size]

        img, gt, wm = self.transforms(img, gt, wm)
        edge = np.where(gt == 2, 1, 0)
        skeleton0 = morphology.skeletonize(edge)
        skeleton0 = np.where(skeleton0 > 0, 1, 0).astype(np.uint8)

        # cmap = 'nipy_spectral'
        # plt.subplot(121)
        # plt.imshow(wm, cmap=plt.get_cmap(cmap))
        # plt.colorbar()
        # plt.subplot(122)
        # plt.imshow(gt, cmap=plt.get_cmap(cmap))
        # plt.colorbar()
        # plt.show()

        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        img = np.array(img, dtype=np.float32)

        img /= 255.

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        skeleton0 = torch.from_numpy(np.array([skeleton0])).float()
        wm = torch.from_numpy(np.array([wm])).float()
        return img, gt, skeleton0, wm


# class balance weight map
def balancewm(mask):
    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [1.0 / np.sum(mask==i) for i in classes ]
    freq /= max(freq)


    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]

    return wc


def distranfwm(mask, beta=4):
    mask = mask.astype('float')
    wc = balancewm(mask)

    dwm = ndimage.distance_transform_edt(mask != 1)
    dwm[dwm > beta] = beta
    dwm = wc + (1.0 - dwm / beta) + 1

    return dwm


if __name__ == '__main__':
    from config import Config
    cfg = Config()
    root = './data/train_images'
    train_dataset = BIPEDDataset(root, crop_size=480)
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0)
    for data_batch in train_loader:
        img, dt = data_batch['images'], data_batch['labels']
        wm = data_batch['weight_mapping']
        print(img.size(), dt.size(), wm.size(),  data_batch['file_name'])
    # crop_size = 400
    # scaleList = [int(crop_size * 0.75),
    #              int(crop_size * 0.875),
    #              crop_size,
    #              int(crop_size * 1.125)]
    # record = []
    # for i in range(1000):
    #     try:
    #         a0 = random.randint(0, len(scaleList))
    #         a = scaleList[a0]
    #     except:
    #         print(f'error {a0}')
    #     if a not in record:
    #         record.append(a)
    # print(record)
