from skimage.transform import resize
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


class Trunc_and_Normalize(object):
    '''
    truncate gray scale and normalize to [0,1]
    '''
    def __init__(self, scale=None):
        self.scale = scale
        if self.scale is not None:
            assert len(self.scale) == 2, 'scale error'

    def __call__(self, sample):
        image = sample['image']
        # gray truncation
        image = image - self.scale[0]
        gray_range = self.scale[1] - self.scale[0]
        image[image < 0] = 0
        image[image > gray_range] = gray_range

        image = image / gray_range

        sample['image'] = image
        return sample


class MRNormalize(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        for i in range(image.shape[0]):
            if np.max(image[i]) != 0:
                image[i] = image[i] / np.max(image[i])

        image[image < 0] = 0

        new_sample = {'image': image, 'label': label}
        return new_sample


class PETandCTNormalize(object):
    def __init__(self, mean=0, w=1024):
        self.mean = mean
        self.w = w

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image[0] = (np.clip(image[0], self.mean - self.w, self.mean + self.w) -
                    self.mean) / self.w
        mean = np.mean(image[1])
        std = np.std(image[1])
        image[1] = (image[1] - mean) / (std + 1e-3)

        new_sample = {'image': image, 'label': label}
        return new_sample


class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[:,crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None, num_class=2, crop=0, channel=1):
        self.dim = dim
        self.num_class = num_class
        self.crop = crop
        self.channel = channel

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        # print(image.dtype)

        mm = 1 if self.channel > 1 else 0
        # crop
        if self.crop != 0:
            if mm:
                image = image[..., self.crop:-self.crop, self.crop:-self.crop]
                label = label[..., self.crop:-self.crop, self.crop:-self.crop]
            else:
                if len(image.shape) == 2:
                    image = image[self.crop:-self.crop, self.crop:-self.crop]
                    label = label[self.crop:-self.crop, self.crop:-self.crop]
                else:
                    image = image[:,self.crop:-self.crop, self.crop:-self.crop]
                    label = label[:,self.crop:-self.crop, self.crop:-self.crop]
        # resize
        if self.dim is not None and label.shape != self.dim:
            if mm:
                for i in range(image.shape[0]):
                    image[i] = resize(image[i], self.dim, anti_aliasing=True)
            else:
                image = resize(image, self.dim, anti_aliasing=True)
            
            temp_label = np.zeros(self.dim, dtype=np.float32)
            for z in range(1, self.num_class):
                roi = resize((label == z).astype(np.float32),
                             self.dim,
                             mode='constant')
                temp_label[roi >= 0.5] = z
            label = temp_label

        new_sample = {'image': image, 'label': label}

        return new_sample


class To_Tensor(object):
    '''
  Convert the data in sample to torch Tensor.
  Args:
  - n_class: the number of class
  '''
    def __init__(self, num_class=2, input_channel=3):
        self.num_class = num_class
        self.channel = input_channel

    def __call__(self, sample):

        image = sample['image']
        label = sample['label']
        
        mm = 1 if self.channel > 1 else 0
        if mm:
            new_image = image[:self.channel,...]
        else:
            new_image = np.expand_dims(image, axis=0)
        new_label = np.empty((self.num_class, ) + label.shape,
                             dtype=np.float32)
        for z in range(1, self.num_class):
            temp = (label == z).astype(np.float32)
            new_label[z, ...] = temp
        new_label[0, ...] = np.amax(new_label[1:, ...], axis=0) == 0

        # convert to Tensor
        new_sample = {
            'image': torch.from_numpy(new_image),
            'label': torch.from_numpy(new_label)
        }

        return new_sample


class DataGenerator(Dataset):
    '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - roi_number: integer or None, to extract the corresponding label
  - num_class: the number of classes of the label
  - transform: the data augmentation methods
  '''
    def __init__(self,
                 path_list,
                 roi_number=None,
                 num_class=2,
                 transform=None,
                 img_key='ct',
                 lab_key='seg'):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.transform = transform
        self.img_key = img_key
        self.lab_key = lab_key

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        image = hdf5_reader(self.path_list[index], self.img_key)
        label = hdf5_reader(self.path_list[index], self.lab_key)
        if self.roi_number is not None:
            if isinstance(self.roi_number, list):
                tmp_mask = np.zeros_like(label, dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[label == roi] = i + 1
                label = tmp_mask
            else:
                assert self.num_class == 2
                label = (label == self.roi_number).astype(np.float32)

        sample = {'image': image, 'label': label}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
