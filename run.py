import os
import argparse
from trainer import SemanticSeg
import random

from config import INIT_TRAINER, CHANNEL, INPUT_SHAPE,SETUP_TRAINER, VERSION, CURRENT_FOLD, PATH_LIST, FOLD_NUM,TEST_PATH,MODE
from utils import get_weight_path,count_params_and_macs
import time

def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    sample_list.sort()
    print('number of sample:',len(sample_list))
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          "Val set length", len(validation_path))
    return train_path, validation_path



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train-cross',
                        choices=["train", 'train-cross',"inf-sw"],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train-cross':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
        print('params and gflops:')
        print(count_params_and_macs(segnetwork.net.cuda(),(1,CHANNEL) + INPUT_SHAPE))
    path_list = PATH_LIST
    # print(len(path_list))
    # Training
    ###############################################
    if args.mode == 'train-cross':
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            print('params and gflops:')
            print(count_params_and_macs(segnetwork.net.cuda(),(1,CHANNEL) + INPUT_SHAPE))
            train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM, current_fold)
            
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM,CURRENT_FOLD)
        
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    elif args.mode == 'inf-sw':
        test_path = TEST_PATH
        # for current_fold in range(1, FOLD_NUM + 1):
        for current_fold in range(1, 6):
            print("=== Predicting Fold ", current_fold, " ===")
            CKPT_PATH = './new_ckpt/{}/{}/fold{}'.format(MODE,VERSION,str(current_fold))
            INIT_TRAINER['weight_path'] = get_weight_path(CKPT_PATH)
            print(INIT_TRAINER['weight_path'])
            segnetwork = SemanticSeg(**INIT_TRAINER)
            save_path = './segout/3d/{}/fold{}'.format(VERSION,current_fold)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # print('test len: %d'%len(pathlist))
            start_time = time.time()
            segnetwork.inference_slidingwindow(test_path,save_path)
            print('run time:%.4f' % (time.time() - start_time))

    ###############################################
