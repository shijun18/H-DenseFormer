import os
import numpy as np
import h5py
from skimage.transform import resize

def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()



def resize_data(input_dir, save_dir, target_size, num_class,img_key='image',lab_key='label'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for sample in os.scandir(input_dir):
        save_path = os.path.join(save_dir,sample.name)
        images = hdf5_reader(sample.path,img_key)
        labels = hdf5_reader(sample.path,lab_key)

        images = resize(images, target_size, mode='constant')
        tmp_labels = np.zeros(target_size, dtype=np.float32)
        for z in range(1,num_class+1):
            roi = resize((labels == z).astype(np.float32),
                        target_size,
                        mode='constant')
            tmp_labels[roi >= 0.5] = z
        labels = tmp_labels

        save_as_hdf5(images.astype(np.int16),save_path,img_key)
        save_as_hdf5(labels.astype(np.uint8),save_path,lab_key)


if __name__ == '__main__':

    input_dir = '/staff/shijun/dataset/Med_Seg/KITS/3d_data'
    save_dir = '../dataset/LITS/3d_data'
    target_size = (256,256,256)
    num_class = 3

    resize_data(input_dir,save_dir,target_size,num_class)