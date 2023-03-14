import os
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import h5py


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

def store_images_labels_2d(save_path, patient_id, cts, labels):

    for i in range(labels.shape[0]):
        ct = cts[:,i,:,:]
        lab = labels[i,:,:]

        hdf5_file = h5py.File(os.path.join(save_path, '%s_%d.hdf5' % (patient_id, i)), 'w')
        hdf5_file.create_dataset('ct', data=ct.astype(np.int16))
        hdf5_file.create_dataset('seg', data=lab.astype(np.uint8))
        hdf5_file.close()


def make_data(csv_file = None):

    base_dir = '../workdir/nnUNet_raw_data/Task2201_picai_baseline/imagesTr'
    label_dir = '../workdir/nnUNet_raw_data/Task2201_picai_baseline/labelsTr'
    hdf5_dir = '../workdir/hdf5_3d_01all'
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    d2_dir = '../workdir/hdf5_2d_01all'
    if not os.path.exists(d2_dir):
        os.makedirs(d2_dir)

    count = 0


    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = list(set(pathlist))
    print(len(pathlist))


    for path in tqdm(pathlist):
        # count += len(sub_path_list)

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))
        seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        seg_image[seg_image>0] = 1
        # if np.max(seg_image) == 0:
        #     continue

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)
        # print(img.shape)

        


        hdf5_path = os.path.join(hdf5_dir, str(count) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(d2_dir,count,img,seg_image)

        # scale_1 = get_scale(in_1)
        # scale_2 = get_scale(in_2)
        # scale_3 = get_scale(in_3)



        count += 1


    print(count)


if __name__ == "__main__":
    make_data()