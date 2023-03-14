import os
import h5py
import numpy as np
import torch


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/1e9))
    print('%.3f M' % (params/1e6))



def get_weight_list(ckpt_path):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    path_list.sort(key=lambda x:x.split('/')[-2])
    return path_list


def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    
    

def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=3):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break  


