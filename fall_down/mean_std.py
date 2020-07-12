# -*- coding: utf-8 -*-
import os
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000
import time


def mean_statistic(folder, channels=3):
    img_paths = []
    for folder_ in folder:
        for imgae_name in sorted(list(os.listdir(folder_))):
            img_paths.append(os.path.join(folder_, imgae_name))
    imgs_num = len(img_paths)
    imgs_mean = [0 for i in range(channels)]

    for path in img_paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        image = image / 255.0
        img_mean = np.mean(image, axis=(0, 1))
        imgs_mean = imgs_mean + img_mean
    imgs_mean = imgs_mean / imgs_num
    return imgs_mean


def std_statistic(folder, channels=3):
    assert isinstance(folder, list)
    img_paths = []
    for folder_ in folder:
        for imgae_name in sorted(list(os.listdir(folder_))):
            img_paths.append(os.path.join(folder_, imgae_name))
    imgs_num = len(img_paths)

    imgs_mean = mean_statistic(folder, channels=3)
    print("mean complete!")

    imgs_std = [0 for i in range(channels)]
    for path in img_paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        image = image / 255.0

        img_std = (image - imgs_mean) ** 2
        img_std = np.mean(img_std, axis=(0, 1))  # [3,]
        imgs_std = imgs_std + img_std
    imgs_std = imgs_std / imgs_num
    imgs_std = np.sqrt(imgs_std)
    return imgs_mean, imgs_std


test_path = ["./B"]  # the dataset which need to compute mean and std
start_time = time.time()
test_mean, test_std =std_statistic(test_path)
print(test_mean, test_std)
print("Total use: ", time.time() - start_time, "s")

# save_txt_path = "./DALC_Fall_Down.txt"
# file = open(save_txt_path, 'w', encoding='utf-8')
# file.writelines('mean£º{}, std£º{}.\n'.format(test_mean, test_std))
# file.writelines('Total use:  {} s.\n'.format(time.time() - start_time))
# file.close()
