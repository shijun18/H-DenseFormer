import os
import glob

from utils import get_weight_path

__2d_net__ = ['unet','unet++','deeplabv3+','pspnet','HDenseFormer_2D_16','HDenseFormer_2D_32']
__encoder_name__ = [None,'resnet18','resnet50']

__3d_net__ = ['unet_3d','da_unet', 'unetr','TransBTS','hecktor20top1','HDenseFormer_32','HDenseFormer_16']
__mode__ = ['3d_seg','2d_seg']

data_path = {
    # competition
    'Hecktor21':'./dataset/Hecktor21/train_3d_seg',
    'PI-CAI22':'./dataset/PI-CAI22/train_2d_seg',
    'LITS':'/staff/shijun/dataset/Med_Seg/LITS/3d_data',
    'KITS':'/staff/shijun/dataset/Med_Seg/KITS/3d_data'
}

channel = {
    'Hecktor21':2,
    'PI-CAI22':3,
    'LITS':1,
    'KITS':1
}

roi_number = {
    'Hecktor21':None,
    'PI-CAI22':None,
    'LITS':2,
    'KITS':2
}

DATASET = 'PI-CAI22'
MODE = '2d_seg'
# MODE = '3d_seg'
NET_NAME = 'HDenseFormer_2D_32'
ENCODER_NAME = None
VERSION = 'v6.6'

DEVICE = '3'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use external pre-trained model 
EX_PRE_TRAINED = False
# True if use resume model
CKPT_POINT = False
CHANNEL = channel[DATASET]

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = roi_number[DATASET] # or 1,2,...
NUM_CLASSES = 2 
ROI_NAME = 'All'
#---------------------------------

#--------------------------------- mode and data path setting
PATH_DIR = data_path[DATASET]
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))
#---------------------------------

#--------------------------------- others
input_shape = {
    'Hecktor21':(144,144,144),
    'PI-CAI22':(384,384),
    'LITS':(448,512,512),
    'KITS':(256,512,512)
}

INPUT_SHAPE = input_shape[DATASET]
BATCH_SIZE = 2 if '3d' in MODE else 24   #48 for resunet else 24(A100) 12(V100) 18(A100 old)


CKPT_PATH = './ckpt/{}/{}/fold{}'.format(MODE,VERSION,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

keys = {
    'Hecktor21':('ct','seg'),
    'PI-CAI22':('ct','seg'),
    'LITS':('image','label'),
    'KITS':('image','label')
}

INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'lr':1e-3, #2d
  'n_epoch':100,
  'channels':CHANNEL,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER, 
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':BATCH_SIZE,
  'num_workers':4,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ex_pre_trained':EX_PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.0001,
  'momentum':0.9,
  'gamma':0.1,
  'milestones':[50,80],
  'T_max':5,
  'topk':10,  
  'use_fp16':False,
  'scale':(-100,200), # for single modality of CT
  'transform_3d':[1,2,4,5,6], # [2,3,4,5,6]
  'transform_2d':[1,6,7,10],  
  'patch_size':(144,144,144),
  'step_size':(72,72,72),
  'transformer_depth':24, #[8,12,24,36]
  'key_touple':keys[DATASET]
 }
#---------------------------------

__loss__ = ['Cross_Entropy','DiceLoss','TopKLoss','CEPlusDice','FocalLoss','FLPlusDice']

LOSS_FUN = 'FocalLoss' if NUM_CLASSES == 2 else 'CEPlusDice'
print('loss is %s'%LOSS_FUN)
SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}/{}'.format(DATASET,MODE,VERSION),
  'log_dir':'./log/{}/{}/{}'.format(DATASET,MODE,VERSION),
  'optimizer':'Adam',
  'loss_fun':LOSS_FUN,
  'class_weight':None,
  'lr_scheduler':'poly_lr',
  'use_ds':'DenseFormer' in NET_NAME
  }
#---------------------------------

TEST_PATH  = None
# SAVE_PATH = './segout/{}_3d/fold{}'.format(VERSION,CURRENT_FOLD)
