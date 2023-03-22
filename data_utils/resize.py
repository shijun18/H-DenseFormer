import os
import numpy as np
import h5py
from skimage.transform import resize
import shutil
import random



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def resize_data(input_dir, save_dir, target_size, num_class, modality=1,img_key='image',lab_key='label'):
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    for sample in os.scandir(input_dir):
        print(sample.name)
        save_path = os.path.join(save_dir,sample.name)
        images = hdf5_reader(sample.path,img_key)
        labels = hdf5_reader(sample.path,lab_key)

        if modality == 1:
            images = resize(images, target_size, mode='constant')
        else:
            images = resize(images, (modality,) + target_size, mode='constant')
        tmp_labels = np.zeros(target_size, dtype=np.float32)
        for z in range(1,num_class+1):
            roi = resize((labels == z).astype(np.float32),
                        target_size,
                        mode='constant')
            tmp_labels[roi >= 0.5] = z
        labels = tmp_labels

        save_as_hdf5(images.astype(np.int16),save_path,img_key)
        save_as_hdf5(labels.astype(np.uint8),save_path,lab_key)


def split_and_resize_data(input_dir, save_dir, target_size, num_class,modality=1,img_key='image',lab_key='label',retain=240):
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    

    sample_list = os.listdir(input_dir)
    random.shuffle(sample_list)

    train_sample = sample_list[:-retain]
    test_sample = sample_list[-retain:]

    train_save_dir = os.path.join(save_dir,'3d_data')
    test_save_dir = os.path.join(save_dir,'3d_test_data')
    
    if os.path.exists(train_save_dir):
        shutil.rmtree(train_save_dir)
    os.makedirs(train_save_dir)

    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
    os.makedirs(test_save_dir)


    for sample in os.scandir(input_dir):
        print(sample.name)
        if sample.name in train_sample:
            save_path = os.path.join(train_save_dir,sample.name)
        else:
            save_path = os.path.join(test_save_dir,sample.name)

        images = hdf5_reader(sample.path,img_key)
        labels = hdf5_reader(sample.path,lab_key)
        print(images.shape,labels.shape)
        if modality == 1:
            images = resize(images, target_size, mode='constant')
        else:
            images = resize(images, (modality,) + target_size, mode='constant')

        tmp_labels = np.zeros(target_size, dtype=np.float32)
        for z in range(1,num_class+1):
            roi = resize((labels == z).astype(np.float32),
                        target_size,
                        mode='constant')
            tmp_labels[roi >= 0.5] = z
        labels = tmp_labels

        save_as_hdf5(images.astype(np.int16),save_path,img_key)
        save_as_hdf5(labels.astype(np.uint8),save_path,lab_key)
        break

if __name__ == '__main__':

    # input_dir = '/staff/shijun/dataset/Med_Seg/BraTS21/npy_data'
    # save_dir = '../dataset/BraTS21'
    # target_size = (128,256,256)
    # num_class = 3
    # modality = 4
    # split_and_resize_data(input_dir,save_dir,target_size,num_class,modality=modality,retain=240)

    input_dir = '/staff/shijun/dataset/Med_Seg/Hecktor21/npy_data'
    save_dir = '../dataset/Hecktor21'
    target_size = (144,144,144)
    num_class = 1
    modality = 2
    split_and_resize_data(input_dir,save_dir,target_size,num_class,img_key='ct',lab_key='seg',modality=modality,retain=44)

    # input_dir = '/staff/shijun/dataset/Med_Seg/KITS/3d_test_data'
    # save_dir = '../dataset/KITS/3d_test_data'
    # target_size = (256,256,256)
    # num_class = 3
    # resize_data(input_dir,save_dir,target_size,num_class)


    # input_dir = '/staff/shijun/dataset/Med_Seg/LITS/3d_test_data'
    # save_dir = '../dataset/LITS/3d_test_data'
    # target_size = (256,256,256)
    # num_class = 2
    # resize_data(input_dir,save_dir,target_size,num_class)

    