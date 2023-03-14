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


class MRNormalize(object):
  def __call__(self,sample):
    ct = sample['ct']
    seg = sample['seg']
    for i in range(ct.shape[0]):
        if np.max(ct[i])!=0:
            ct[i] = ct[i]/np.max(ct[i])
        
    ct[ct<0] = 0

    new_sample = {'ct':ct, 'seg':seg}
    return new_sample

class PETandCTNormalize(object):
  def __init__(self,mean=0,w=1024):
      self.mean = mean
      self.w = w

  def __call__(self,sample):
    ct = sample['ct']
    seg = sample['seg']
    ct[0] = (np.clip(ct[0], self.mean - self.w, self.mean + self.w) - self.mean)/ self.w
    mean = np.mean(ct[1])
    std = np.std(ct[1])
    ct[1] = (ct[1] - mean) / (std + 1e-3)

    new_sample = {'ct':ct, 'seg':seg}
    return new_sample



class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[:,crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None, num_class=2, crop=0):
        self.dim = dim
        self.num_class = num_class
        self.crop = crop

    def __call__(self, sample):
        ct = sample['ct']
        seg = sample['seg']
        # print(image.dtype)

        # crop
        if self.crop != 0:
            if len(seg.shape) > 2:
                ct = ct[:,self.crop:-self.crop, self.crop:-self.crop]
                seg = seg[:,self.crop:-self.crop, self.crop:-self.crop]
            else:
                ct = ct[self.crop:-self.crop, self.crop:-self.crop]
                seg = seg[self.crop:-self.crop, self.crop:-self.crop]
        # resize
        if self.dim is not None and seg.shape != self.dim:
            for i in range(ct.shape[0]):
                ct[i] = resize(ct[i], self.dim, anti_aliasing=True)
            temp_label = np.zeros(self.dim,dtype=np.float32)
            for z in range(1, self.num_class):
                roi = resize((seg == z).astype(np.float32),self.dim,mode='constant')
                temp_label[roi >= 0.5] = z
            seg = temp_label
        
        new_sample = { 'ct': ct, 'seg':seg}

        return new_sample



class To_Tensor(object):
  '''
  Convert the data in sample to torch Tensor.
  Args:
  - n_class: the number of class
  '''
  def __init__(self,num_class=2,input_channel = 3):
    self.num_class = num_class
    self.channel = input_channel

  def __call__(self,sample):

    ct = sample['ct']
    seg = sample['seg']
    # new_image = image.transpose(2,0,1)
    # print(image.shape)
    # expand dims
    # print(ct.shape)
    # print(seg.shape)

    new_image = ct[:self.channel,...]
    new_label = np.empty((self.num_class,) + seg.shape, dtype=np.float32)
    for z in range(1, self.num_class):
        temp = (seg==z).astype(np.float32)
        new_label[z,...] = temp
    new_label[0,...] = np.amax(new_label[1:,...],axis=0) == 0   
   
    # convert to Tensor
    new_sample = {'image': torch.from_numpy(new_image),
                  'label': torch.from_numpy(new_label)}
    
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
  def __init__(self, path_list, roi_number=None, num_class=2, transform=None):

    self.path_list = path_list
    self.roi_number = roi_number
    self.num_class = num_class
    self.transform = transform


  def __len__(self):
    return len(self.path_list)


  def __getitem__(self,index):
    ct = hdf5_reader(self.path_list[index],'ct')
    seg = hdf5_reader(self.path_list[index],'seg')
    if self.roi_number is not None:
        if isinstance(self.roi_number,list):
            tmp_mask = np.zeros_like(seg,dtype=np.float32)
            assert self.num_class == len(self.roi_number) + 1
            for i, roi in enumerate(self.roi_number):
                tmp_mask[seg == roi] = i+1
            seg = tmp_mask
        else:
          assert self.num_class == 2
          seg = (seg==self.roi_number).astype(np.float32) 

    sample = {'ct': ct, 'seg':seg}
    # Transform
    if self.transform is not None:
      sample = self.transform(sample)

    return sample

