import os
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py
import shutil
from skimage.transform import resize
import random

def get_info(data):
    info = []
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

def get_scale(data):
    info = []
    info.append(np.mean(data))
    info.append(np.max(data))
    info.append(np.min(data))
    info.append(np.std(data))
    return info

def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image

def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def csv_reader_single(csv_file,key_col=None,value_col=None):
    '''
    Extracts the specified single column, return a single level dict.
    The value of specified column as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique. 
    - value_col: string,  specified column as value
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = file_csv[value_col].values.tolist()
    
    target_dict = {}
    for key_item,value_item in zip(key_list,value_list):
        target_dict[key_item] = value_item

    return target_dict

def store_images_labels_2d(save_path, patient_id, cts, labels):

    for i in range(labels.shape[0]):
        ct = cts[:,i,:,:]
        lab = labels[i,:,:]

        hdf5_file = h5py.File(os.path.join(save_path, '%s_%d.hdf5' % (patient_id, i)), 'w')
        hdf5_file.create_dataset('ct', data=ct.astype(np.int16))
        hdf5_file.create_dataset('seg', data=lab.astype(np.uint8))
        hdf5_file.close()

def make_valdata():
    hdf5_dir = './dataset/hdf5_3d_all_removering'
    out_dir = './dataset/hdf5_3d_all_removering_normalize'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in tqdm(list(os.listdir(hdf5_dir))):
        ct = hdf5_reader(os.path.join(hdf5_dir,file),'ct')
        seg = hdf5_reader(os.path.join(hdf5_dir,file),'seg')
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if np.max(ct[i,j])!=0:
                    ct[i,j] = ct[i,j]/np.max(ct[i,j])
        hdf5_path = os.path.join(out_dir, file)

        save_as_hdf5(ct,hdf5_path,'ct')
        save_as_hdf5(seg,hdf5_path,'seg')

def make_data(csv_file = None):

    base_dir = '../workdir/nnUNet_test_data_5m'
    label_dir = './segout/vote_all_removering'
    hdf5_dir = './dataset/hdf5_3d_all_removering_3c_5m'
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    d2_dir = './dataset/hdf5_2d_all_removering_3c_5m'
    if not os.path.exists(d2_dir):
        os.makedirs(d2_dir)

    csv_path = '/staff/honeyk/project/XunFei_Classifier-main/dataset/picai/test3c.csv'
    label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')

    count = 0

    rand_list = list(range(1500))
    random.shuffle(rand_list)
    print(rand_list)

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = list(set(pathlist))
    l = len(pathlist)
    print(l)

    zs = 0
    c1,c2=0,0

    for i in tqdm(range(l)):
        # count += len(sub_path_list)
        path = pathlist[i]
        # break

        seg_image = np.load(os.path.join(label_dir,path + '.npy')).astype(np.uint8)
        if np.sum(seg_image) <= 20:
            print(path)
            zs+=1
            # count += 1
            # continue

        seg_image *= int(label_dict[path])
        if int(label_dict[path])==1:
            c1+=1
        if int(label_dict[path])==2:
            c2+=1
        
        # print(label_dict[path])

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))
        in_4 = sitk.ReadImage(os.path.join(base_dir,path + '_0003.nii.gz'))
        in_5 = sitk.ReadImage(os.path.join(base_dir,path + '_0004.nii.gz'))
        

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        in_4 = sitk.GetArrayFromImage(in_4).astype(np.int16)
        in_5 = sitk.GetArrayFromImage(in_5).astype(np.int16)
        img = np.stack((in_1,in_2,in_3,in_4,in_5),axis=0)
        # print(img.shape)

        outc = rand_list[count]


        hdf5_path = os.path.join(hdf5_dir, str(outc) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(d2_dir,outc,img,seg_image)

        count += 1

    print(zs)
    
    base_dir = '../nnUNet_raw_data/Task2203_picai_baseline/imagesTr'
    label_dir = '../nnUNet_raw_data/Task2203_picai_baseline/labelsTr'


    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = list(set(pathlist))
    print(len(pathlist))


    for path in tqdm(pathlist):
        # count += len(sub_path_list)

        seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        seg_image[seg_image==2] = 1
        seg_image[seg_image>2] = 2
        if np.max(seg_image) == 1:
            c1+=1
        if np.max(seg_image) == 2:
            c2+=1
        # print(np.max(seg_image))
        # if np.max(seg_image) == 0:
        #     continue

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))
        in_4 = sitk.ReadImage(os.path.join(base_dir,path + '_0003.nii.gz'))
        in_5 = sitk.ReadImage(os.path.join(base_dir,path + '_0004.nii.gz'))
        
        # print(np.sum(seg_image))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        in_4 = sitk.GetArrayFromImage(in_4).astype(np.int16)
        in_5 = sitk.GetArrayFromImage(in_5).astype(np.int16)
        img = np.stack((in_1,in_2,in_3,in_4,in_5),axis=0)
        # print(img.shape)

        outc = rand_list[count]


        hdf5_path = os.path.join(hdf5_dir, str(outc) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(d2_dir,outc,img,seg_image)

        count += 1


    print(count)
    print(c1,c2)

if __name__ == "__main__":
    make_data()
    # make_valdata()