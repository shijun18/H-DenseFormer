import os
import numpy as np
import h5py
from skimage.transform import resize
import SimpleITK as sitk
import shutil

def nii_reader(data_path):
    data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    return data,image


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def nii2npy(input_dir,save_dir,img_key='image',lab_key='label'):

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    for sample in os.scandir(input_dir):
        ct_path = os.path.join(sample.path,f'{sample.name}_ct.nii.gz')
        pet_path = os.path.join(sample.path,f'{sample.name}_pt.nii.gz')
        mask_path = os.path.join(sample.path,f'{sample.name}_ct_gtvt.nii.gz')

        _,ct = nii_reader(ct_path)
        _,pet = nii_reader(pet_path)
        _,mask = nii_reader(mask_path)

        images = np.asarray([ct,pet]).astype(np.int16)
        labels = mask.astype(np.uint8)

        print(images.shape,labels.shape)
        print(np.unique(labels))

        save_path = os.path.join(save_dir,sample.name + '.hdf5')

        save_as_hdf5(images,save_path,img_key)
        save_as_hdf5(labels,save_path,lab_key)


if __name__ == '__main__':
    
    input_dir = '/staff/shijun/dataset/Hecktor21/resampled_nii_data'
    save_dir = '/staff/shijun/dataset/Med_Seg/Hecktor21/npy_data'

    nii2npy(input_dir,save_dir)
