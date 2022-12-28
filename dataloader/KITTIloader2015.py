import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = os.path.join(filepath, 'image_2/')
  right_fold = os.path.join(filepath, 'image_3/')
  disp_L = os.path.join(filepath, 'disp_occ_0/')
  disp_R = os.path.join(filepath, 'disp_occ_1/')

  image = [img for img in os.listdir(left_fold) if img.find('_10') > -1]

  train = image[:160]
  val   = image[160:]

  left_train  = [os.path.join(left_fold, img) for img in train]
  right_train = [os.path.join(right_fold, img) for img in train]
  disp_train_L = [os.path.join(disp_L, img) for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = [os.path.join(left_fold, img) for img in val]
  right_val = [os.path.join(right_fold, img) for img in val]
  disp_val_L = [os.path.join(disp_L, img) for img in val]
  #disp_val_R = [filepath+disp_R+img for img in val]

  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L