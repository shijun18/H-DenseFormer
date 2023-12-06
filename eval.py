import glob
import os 
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformer_2d import RandomFlip2D, RandomRotate2D, RandomErase2D,RandomZoom2D,RandomAdjust2D,RandomNoise2D,RandomDistort2D
from data_loader import DataGenerator, To_Tensor, CropResize, Normalize
from torch.cuda.amp import autocast as autocast
from utils import get_weight_path
from converter import hdf5_reader
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm

def get_net(net_name,encoder_name,channels=1,num_classes=2,input_shape=(512,512),transformer_depth=24):

        if net_name == 'unet':
            if encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=channels,
                    classes=num_classes,                     
                    aux_params={"classes":num_classes-1} 
                )
        elif net_name == 'unet++':
            if encoder_name is None:
                raise ValueError(
                    "encoder name must not be 'None'!"
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.UnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=channels,
                    classes=num_classes,                     
                    aux_params={"classes":num_classes-1} 
                )
        
        elif net_name == 'deeplabv3+':
            if encoder_name is None:
                raise ValueError(
                    "encoder name must not be 'None'!"
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=channels,
                    classes=num_classes,                     
                    aux_params={"classes":num_classes-1} 
                )
        
        elif net_name == 'HDenseFormer_32':
            from models.HDenseFormer import HDenseFormer_32
            net = HDenseFormer_32(in_channels=channels,
                                  n_cls=num_classes,
                                  image_size=input_shape,
                                  transformer_depth=transformer_depth)

        elif net_name == 'HDenseFormer_16':
            from models.HDenseFormer import HDenseFormer_16
            net = HDenseFormer_16(in_channels=channels,
                                  n_cls=num_classes,
                                  image_size=input_shape,
                                  transformer_depth=transformer_depth)

        elif net_name == 'hecktor20top1':
            from models.Hecktor20Top1.model import hecktertop1
            net = hecktertop1(in_channels=channels,
                              n_cls=num_classes)

        elif net_name == 'TransBTS':
            from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
            _, net = TransBTS(n_channels=channels,
                              num_classes=num_classes,
                              img_dim=input_shape[0], 
                              _conv_repr=True, 
                              _pe_type="learned")

        elif net_name == 'da_unet':
            from models.DAUNet import da_unet
            net = da_unet(init_depth=input_shape[0],
                          n_channels=channels,
                          n_classes=num_classes)

        elif net_name == 'unetr':
            from models.UNETR import UNETR
            net = UNETR(in_channels=channels,
                        out_channels=num_classes,
                        img_size=tuple(input_shape),
                        feature_size=16,
                        hidden_size=768,
                        mlp_dim=3072,
                        num_heads=12,
                        pos_embed='perceptron',
                        norm_name='instance',
                        conv_block=True,
                        res_block=True,
                        dropout_rate=0.0)
        
        return net

class Normalize_2d(object):
    def __call__(self,sample):
        ct = sample['ct']
        seg = sample['seg']
        # print(np.max(ct[0]),np.max(ct[1]),np.max(ct[2]))
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if np.max(ct[i,j])!=0:
                    ct[i,j] = ct[i,j]/np.max(ct[i,j])
            
        new_sample = {'ct':ct, 'seg':seg}
        return new_sample

def eval_process(test_path,config):
    # data loader
    test_transformer = transforms.Compose([
                Normalize_2d(),
                # CropResize(dim=config.input_shape,num_class=config.num_classes,crop=config.crop),
                To_Tensor(num_class=config.num_classes)
            ])

    ct = hdf5_reader(test_path,'ct')
    seg = hdf5_reader(test_path,'seg')
    sample = {'ct': ct, 'seg':seg}
    sample = test_transformer(sample)

    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    # weight_path = os.path.join(config.ckpt_path,'last.pth')
    print(weight_path)

    # get net
    net = get_net(config.net_name,config.encoder_name,config.channels,config.num_classes,config.input_shape,config.transformer_depth)
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['state_dict'])

    pred = []
    true = []
    net = net.cuda()
    net.eval()

    with torch.no_grad():
        # for step, sample in enumerate(test_loader):
            data = sample['image']
            target = sample['label']

            data = data.squeeze().transpose(1,0) 

            data = data.cuda()

            with autocast(False):
                output = net(data)
                if isinstance(output,tuple):
                    output = output[0]

            if 'nnloss' in config.net_name:
                output = output[0]

            seg_output = torch.argmax(torch.softmax(output, dim=1),1).detach().cpu().numpy()                          
            # target = torch.argmax(target,1).detach().cpu().numpy()
            # pred.append(seg_output)
            # true.append(target)
    # pred = np.concatenate(pred,axis=0)
    # true = np.concatenate(true,axis=0)

    return seg_output

def predict_process(test_path,config,base_dir):
    # test_transformer = transforms.Compose([
    #         Normalize_2d(),
    #         To_Tensor(num_class=config.num_classes)
    #     ])

    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    # get net
    net = get_net(config.net_name,config.encoder_name,config.channels,config.num_classes,config.input_shape)
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['state_dict'])

    pred = []
    net = net.cuda()
    net.eval()

    in_1 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0000.nii.gz'))
    in_2 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0001.nii.gz'))
    in_3 = sitk.ReadImage(os.path.join(base_dir,test_path + '_0002.nii.gz'))

    in_1 = sitk.GetArrayFromImage(in_1).astype(np.float32)
    in_2 = sitk.GetArrayFromImage(in_2).astype(np.float32)
    in_3 = sitk.GetArrayFromImage(in_3).astype(np.float32)

    image = np.stack((in_1,in_2,in_3),axis=0)

    with torch.no_grad():
        for i in range(image.shape[1]):
            new_image = image[:,i,:,:]
            for j in range(new_image.shape[0]):
                if np.max(new_image[j]) != 0:
                    new_image[j] = new_image[j]/np.max(new_image[j])
            new_image = np.expand_dims(new_image,axis=0)
            data = torch.from_numpy(new_image)

            data = data.cuda()
            with autocast(False):
                output = net(data)
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output  
            # print(seg_output.size())
            seg_output = torch.softmax(seg_output, dim=1).detach().cpu().numpy()   
            pred.append(seg_output) 
    
    pred = np.concatenate(pred,axis=0).transpose((1,0,2,3))
    print(np.sum(pred[1]))
    return pred



def save_npy(data_path):
    config = Config()
    for fold in range(1,6):
        print('****fold%d****'%fold)
        config.fold = fold
        config.ckpt_path = f'./new_ckpt/2d_seg/{config.version}/fold{str(fold)}'
        # print(get_weight_path(config.ckpt_path))
        save_dir = f'./segout/2d/{config.version}/fold{str(fold)}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # path_list = glob.glob(os.path.join(data_path,'*.nii.gz'))
        # pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(data_path)]
        # pathlist = list(set(pathlist))
        pathlist = glob.glob(os.path.join(data_path,'*.hdf5'))
        print(len(pathlist))
        # label_dir = '../workdir/nnUNet_raw_data/Task2201_picai_baseline/labelsTr'
        for path in pathlist:
            # seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

            # seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
            # seg_image[seg_image>0] = 1
            # if np.max(seg_image) == 0:
            #     continue
            # print(path)
            pred = eval_process(path,config)
            # print(pred.shape)
            print(np.sum(pred))
            np.save(os.path.join(save_dir,path.split('/')[-1].split('.')[0]+'.npy'),pred)


        # print("runtime:%.3f"%(time.time() - start))

def save_nii(data_path):
    config = Config()
    for fold in range(1,6):
        print('****fold%d****'%fold)
        config.fold = fold
        config.ckpt_path = f'./new_ckpt/2d/{config.version}/fold{str(fold)}'
        # print(get_weight_path(config.ckpt_path))
        save_dir = f'./segout/2d/{config.version}_2d/fold{str(fold)}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path_list = glob.glob(os.path.join(data_path,'*.nii.gz'))
        pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(data_path)]
        pathlist = list(set(pathlist))
        print(len(pathlist))
        for path in path_list:
            print(path)
            pred = predict_process(path,config)
            out = sitk.GetImageFromArray(pred)
            sitk.WriteImage(out,os.path.join(save_dir,path + '.nii.gz'))
            
class Config:
    
    input_shape = (384,384)
    channels = 3
    crop = 0
    roi_number = None
    num_classes = 2
    transformer_depth = 12
    # net_name = 'res_unet'
    # # encoder_name = None
    # encoder_name = 'resnet18'
    # version = 'v13.0'
    net_name = 'pspnet'
    encoder_name = None
    version = 'v27.0'
    fold = 1
    ckpt_path = f'./new_ckpt/2d/{version}/fold{str(fold)}'

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # test data
    data_path = './dataset/test_3d_seg'
    save_npy(data_path)
