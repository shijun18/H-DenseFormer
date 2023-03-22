import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import glob
import shutil
from utils import dfs_remove_weight, hdf5_reader
from scipy.ndimage.filters import gaussian_filter

from torch.nn import functional as F

from data_utils.transformer_2d import RandomFlip2D, RandomRotate2D, RandomErase2D, RandomAdjust2D, RandomDistort2D, RandomZoom2D, RandomNoise2D
from data_utils.data_loader import DataGenerator, CropResize, To_Tensor, PETandCTNormalize, MRNormalize,Trunc_and_Normalize
from data_utils.transformer_3d import RandomTranslationRotationZoom3D, RandomCrop3D, RandomFlip3D
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import setproctitle
import warnings
warnings.filterwarnings('ignore')

# GPU version.


class SemanticSeg(object):
    '''
  Control the training, evaluation, and inference process.
  Args:
  - net_name: string
  - lr: float, learning rate.
  - n_epoch: integer, the epoch number
  - channels: integer, the channel number of the input
  - num_classes: integer, the number of class
  - input_shape: tuple of integer, input dim
  - crop: integer, cropping size
  - batch_size: integer
  - num_workers: integer, how many subprocesses to use for data loading.
  - device: string, use the specified device
  - pre_trained: True or False, default False
  - weight_path: weight path of pre-trained model
  '''
    def __init__(self,
                 net_name=None,
                 encoder_name=None,
                 lr=1e-3,
                 n_epoch=1,
                 channels=1,
                 num_classes=2,
                 roi_number=1,
                 scale=None,
                 input_shape=None,
                 crop=48,
                 batch_size=6,
                 num_workers=0,
                 device=None,
                 pre_trained=False,
                 ex_pre_trained=False,
                 ckpt_point=True,
                 weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 topk=50,
                 use_fp16=True,
                 transform_3d=None,
                 transform_2d=None,
                 patch_size=(128, 256, 256),
                 step_size=(64, 128, 128),
                 transformer_depth=18,
                 key_touple=('ct','seg')):
        super(SemanticSeg, self).__init__()

        self.net_name = net_name
        self.encoder_name = encoder_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.roi_number = roi_number
        self.scale = scale
        self.input_shape = input_shape
        self.crop = crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.ex_pre_trained = ex_pre_trained
        self.ckpt_point = ckpt_point
        self.weight_path = weight_path

        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metrics_threshold = 0.

        self.max_epoch = 1000

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.topk = topk
        self.use_fp16 = use_fp16

        self.patch_size = patch_size
        self.step_size = step_size
        self.transformer_depth = transformer_depth
        self.key_touple = key_touple

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_net(self.net_name)

        if self.pre_trained:
            self._get_pre_trained(self.weight_path, ckpt_point)

        if self.roi_number is not None:
            assert self.num_classes == 2, "num_classes must be set to 2 for binary segmentation"

        self.transform_list_3d = [
            RandomCrop3D(self.patch_size),  #1
            PETandCTNormalize(),  #2
            CropResize(dim=self.input_shape,
                       num_class=self.num_classes,
                       crop=self.crop,
                       channel=self.channels),  #3
            RandomTranslationRotationZoom3D(mode='tr',
                                            num_class=self.num_classes),  #4
            RandomFlip3D(mode='hv'),  #5
            To_Tensor(num_class=self.num_classes,
                      input_channel=self.channels),  #6
            Trunc_and_Normalize(scale=self.scale), #7
            MRNormalize(),  #1
        ]

        self.train_transform_3d = [
            self.transform_list_3d[i - 1] for i in transform_3d
        ]
        self.val_transform_3d = [
            self.transform_list_3d[i - 1] for i in transform_3d
            if i in [1, 2, 3, 6]
        ]

        self.transform_list_2d = [
            MRNormalize(),  #1
            CropResize(dim=self.input_shape,
                       num_class=self.num_classes,
                       crop=self.crop,
                       channel=self.channels),  #2
            RandomErase2D(scale_flag=False),  #3
            RandomZoom2D(),  #4
            RandomDistort2D(),  #5
            RandomRotate2D(),  #6
            RandomFlip2D(mode='hv'),  #7
            RandomAdjust2D(),  #8
            RandomNoise2D(),  # 9
            To_Tensor(num_class=self.num_classes,
                      input_channel=self.channels),  # 10
            Trunc_and_Normalize(scale=self.scale) #11
        ]

        self.train_transform_2d = [
            self.transform_list_2d[i - 1] for i in transform_2d
        ]
        self.val_transform_2d = [
            self.transform_list_2d[i - 1] for i in transform_2d
            if i in [1, 2, 10]
        ]

    def trainer(self,
                train_path,
                val_path,
                cur_fold,
                output_dir=None,
                log_dir=None,
                optimizer='Adam',
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None,
                use_ds=False):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold" + str(cur_fold))
        log_dir = os.path.join(log_dir, "fold" + str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        self.step_pre_epoch = len(train_path) // self.batch_size
        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(
            len(train_path) / self.batch_size)
        
        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)

        if use_ds:
            from loss.combine_loss import DeepSuperloss
            loss = DeepSuperloss(criterion=loss)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        if len(self.input_shape) > 2:
            train_transformer = transforms.Compose(self.train_transform_3d)
        else:
            train_transformer = transforms.Compose(self.train_transform_2d)

        train_dataset = DataGenerator(train_path,
                                      roi_number=self.roi_number,
                                      num_class=self.num_classes,
                                      transform=train_transformer,
                                      img_key=self.key_touple[0],
                                      lab_key=self.key_touple[1])

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr)

        scaler = GradScaler()

        # if self.pre_trained and self.ckpt_point:
        #     checkpoint = torch.load(self.weight_path)
        # optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        early_stopping = EarlyStopping(patience=30,
                                       verbose=True,
                                       monitor='val_dice',
                                       op_type='max')
 
        for epoch in range(self.start_epoch, self.n_epoch):
            setproctitle.setproctitle('{}: {}/{}'.format(self.net_name, epoch, self.n_epoch))

            train_loss, train_dice, train_run_dice = self._train_on_epoch(
                epoch, net, loss, optimizer, train_loader, scaler)

            val_loss, val_dice, val_run_dice = self._val_on_epoch(
                epoch, net, loss, val_path)

            if lr_scheduler is not None:
                lr_scheduler.step()

            torch.cuda.empty_cache()

            print('epoch:{}/{},train_loss:{:.5f},val_loss:{:.5f}'.format(
                epoch, self.n_epoch, train_loss, val_loss))

            print(
                'epoch:{}/{},train_dice:{:.5f},train_run_dice:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f}'
                .format(epoch, self.n_epoch, train_dice, train_run_dice,
                        val_dice, val_run_dice))

            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/dice', {
                'train': train_dice,
                'val': val_dice
            }, epoch)
            self.writer.add_scalars('data/run_dice', {
                'train': train_run_dice,
                'val': val_run_dice
            }, epoch)

            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'],
                                   epoch)

            early_stopping(val_dice)

            #save
            # if val_run_dice > self.metrics_threshold:
            #     self.metrics_threshold = val_run_dice
            if val_dice > self.metrics_threshold:
                self.metrics_threshold = val_dice
                # if score > self.metrics_threshold:
                #     self.metrics_threshold = score

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    # 'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch={}-train_loss={:.5f}-train_dice:={:.5f}-train_run_dice={:.5f}-val_loss={:.5f}-val_dice={:.5f}-val_run_dice={:.5f}.pth'.format(
                    epoch, train_loss, train_dice, train_run_dice, val_loss,
                    val_dice, val_run_dice)

                save_path = os.path.join(output_dir, file_name)
                print("Save as: %s" % file_name)

                torch.save(saver, save_path)

            epoch += 1

            # early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()
        dfs_remove_weight(output_dir, retain=3)

    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader,
                        scaler):

        net.train()

        train_loss = AverageMeter()
        train_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            with autocast(self.use_fp16):
                output = net(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if isinstance(output, list):
                output = output[0]

            output = output.float()
            loss = loss.float()

            # measure dice and record loss
            dice = compute_dice(output.detach(), target)
            train_loss.update(loss.item(), data.size(0))
            train_dice.update(dice.item(), data.size(0))

            # measure run dice
            output = torch.argmax(torch.softmax(output, dim=1),
                                1).detach().cpu().numpy()  #N*H*W

            target = torch.argmax(target, 1).detach().cpu().numpy()
            run_dice.update_matrix(target, output)

            torch.cuda.empty_cache()

            if self.global_step % 10 == 0:
                rundice, dice_list = run_dice.compute_dice()
                print("Category Dice: ", dice_list)
                print(
                    'epoch:{}/{},step:{},train_loss:{:.5f},train_dice:{:.5f},run_dice:{:.5f},lr:{}'
                    .format(epoch, self.n_epoch, step, loss.item(),
                            dice.item(), rundice,
                            optimizer.param_groups[0]['lr']))
                # run_dice.init_op()
                self.writer.add_scalars('data/train_loss_dice', {
                    'train_loss': loss.item(),
                    'train_dice': dice.item()
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_dice.avg, run_dice.compute_dice()[0]

    def _val_on_epoch(self, epoch, net, criterion, val_path):

        net.eval()

        if len(self.input_shape) > 2:
            val_transformer = transforms.Compose(self.val_transform_3d)
        else:
            val_transformer = transforms.Compose(self.val_transform_2d)

        val_dataset = DataGenerator(val_path,
                                    roi_number=self.roi_number,
                                    num_class=self.num_classes,
                                    transform=val_transformer,
                                    img_key=self.key_touple[0],
                                    lab_key=self.key_touple[1])

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        val_loss = AverageMeter()
        val_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                loss = criterion(output, target)

                if isinstance(output, list):
                    output = output[0]

                output = output.float()
                loss = loss.float()

                # measure dice and record loss
                dice = compute_dice(output.detach(), target)
                val_loss.update(loss.item(), data.size(0))
                val_dice.update(dice.item(), data.size(0))

                output = torch.softmax(output, dim=1)
                output = torch.argmax(output, 1).detach().cpu().numpy()  #N*H*W
                target = torch.argmax(target, 1).detach().cpu().numpy()
                run_dice.update_matrix(target, output)

                torch.cuda.empty_cache()

                if step % 10 == 0:
                    rundice, dice_list = run_dice.compute_dice()
                    print("Category Dice: ", dice_list)
                    print(
                        'epoch:{}/{},step:{},val_loss:{:.5f},val_dice:{:.5f},run_dice:{:.5f}'
                        .format(epoch, self.n_epoch, step, loss.item(),
                                dice.item(), rundice))
                    # run_dice.init_op()

        return val_loss.avg, val_dice.avg, run_dice.compute_dice()[0]

    def inference_slidingwindow(self, test_path, save_path, net=None):

        if net is None:
            net = self.net

        net = net.cuda()
        net.eval()

        pathlist = glob.glob(os.path.join(test_path, '*.hdf5'))

        test_transformer = transforms.Compose([
            PETandCTNormalize(),  #2
            To_Tensor(num_class=self.num_classes)  #6
        ])

        patch_size = self.patch_size

        with torch.no_grad():
            for step, path in enumerate(pathlist):
                print(path)

                image = hdf5_reader(path, 'ct')
                label = hdf5_reader(path, 'label')
                sample = {'image': image, 'label': label}

                # Transform
                if test_transformer is not None:
                    sample = test_transformer(sample)

                ori_image = np.asarray(sample['image'])

                new_image = np.expand_dims(ori_image, axis=0)

                aggregated_results = torch.zeros(
                    [1, self.num_classes] +
                    list(new_image.shape[2:]), ).cuda()
                aggregated_nb_of_predictions = torch.zeros(
                    [1, self.num_classes] +
                    list(new_image.shape[2:]), ).cuda()

                steps = self.cal_steps(ori_image.shape[1:])

                for x in steps[0]:
                    lb_x = x
                    ub_x = x + patch_size[0] if x + patch_size[
                        0] <= ori_image.shape[1] else ori_image.shape[1]
                    for y in steps[1]:
                        lb_y = y
                        ub_y = y + patch_size[1] if y + patch_size[
                            1] <= ori_image.shape[2] else ori_image.shape[2]
                        for z in steps[2]:
                            lb_z = z
                            ub_z = z + patch_size[2] if z + patch_size[
                                2] <= ori_image.shape[3] else ori_image.shape[3]

                            image = ori_image[:, lb_x:ub_x, lb_y:ub_y,
                                              lb_z:ub_z]

                            image = np.expand_dims(image, axis=0)

                            data = torch.from_numpy(image).float()
                            data = data.cuda()

                            with autocast(self.use_fp16):
                                predicted_patch = net(data)
                                if isinstance(predicted_patch, tuple):
                                    predicted_patch = predicted_patch[0]

                            if isinstance(predicted_patch, list):
                                predicted_patch = predicted_patch[0]
                            predicted_patch = predicted_patch.float()  #N*C

                            predicted_patch = F.softmax(predicted_patch, dim=1)
                            predicted_patch = F.interpolate(
                                predicted_patch,
                                (ub_x - lb_x, ub_y - lb_y, ub_z - lb_z))
                            # print(predicted_patch.size()[2:])

                            gaussian_importance_map = torch.from_numpy(
                                self.get_gaussian(
                                    np.asarray(
                                        predicted_patch.size()[2:]))).cuda()
                            aggregated_results[:, :, lb_x:ub_x, lb_y:ub_y,
                                               lb_z:
                                               ub_z] += predicted_patch  #* gaussian_importance_map
                            aggregated_nb_of_predictions[:, :, lb_x:ub_x,
                                                         lb_y:ub_y,
                                                         lb_z:ub_z] += 1
                            # aggregated_nb_of_predictions[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += gaussian_importance_map

                output = aggregated_results / aggregated_nb_of_predictions

                # measure run dice
                # output = output.detach().cpu().numpy().squeeze()
                output = torch.argmax(torch.softmax(
                    output, dim=1), 1).detach().cpu().numpy().squeeze()  #N*H*W

                print(output.shape)
                print(np.sum(output))

                np.save(
                    os.path.join(save_path,
                                 path.split('/')[-1].split('.')[0] + '.npy'),
                    output)

                torch.cuda.empty_cache()

    def cal_steps(self, image_size):
        patch_size = self.patch_size
        step_size = self.step_size

        steps = []

        for dim in range(len(image_size)):
            if image_size[dim] <= patch_size[dim]:
                steps_here = [
                    0,
                ]
            else:
                max_step_value = image_size[dim] - patch_size[dim]
                num_steps = int(np.ceil((max_step_value) / step_size[dim])) + 1
                actual_step_size = max_step_value / (num_steps - 1)
                steps_here = [
                    int(np.round(actual_step_size * i))
                    for i in range(num_steps)
                ]

            steps.append(steps_here)

        # print(steps)
        return steps

    def get_gaussian(self, patch_size, sigma_scale=1. / 8):
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp,
                                                  sigmas,
                                                  0,
                                                  mode='constant',
                                                  cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(
            gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    def _get_net(self, net_name):

        if net_name == 'HDenseFormer_32':
            from models.HDenseFormer import HDenseFormer_32
            net = HDenseFormer_32(in_channels=self.channels,
                                  n_cls=self.num_classes,
                                  image_size=self.input_shape,
                                  transformer_depth=self.transformer_depth)

        elif net_name == 'HDenseFormer_16':
            from models.HDenseFormer import HDenseFormer_16
            net = HDenseFormer_16(in_channels=self.channels,
                                  n_cls=self.num_classes,
                                  image_size=self.input_shape,
                                  transformer_depth=self.transformer_depth)
        
        elif net_name == 'HDenseFormer_2D_32':
            from models.HDenseFormer_2D import HDenseFormer_2D_32
            net = HDenseFormer_2D_32(in_channels=self.channels,
                                  n_cls=self.num_classes,
                                  image_size=self.input_shape,
                                  transformer_depth=self.transformer_depth)

        elif net_name == 'HDenseFormer_2D_16':
            from models.HDenseFormer_2D import HDenseFormer_2D_16
            net = HDenseFormer_2D_16(in_channels=self.channels,
                                  n_cls=self.num_classes,
                                  image_size=self.input_shape,
                                  transformer_depth=self.transformer_depth)

        elif net_name == 'hecktor20top1':
            from models.Hecktor20Top1.model import hecktertop1
            net = hecktertop1(in_channels=self.channels,
                              n_cls=self.num_classes)

        elif net_name == 'TransBTS':
            from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
            _, net = TransBTS(n_channels=self.channels,
                              num_classes=self.num_classes,
                              img_dim=self.input_shape[0], 
                              _conv_repr=True, 
                              _pe_type="learned")

        elif net_name == 'da_unet':
            from models.DAUNet import da_unet
            net = da_unet(init_depth=self.input_shape[0],
                          n_channels=self.channels,
                          n_classes=self.num_classes)

        elif net_name == 'unetr':
            from models.UNETR import UNETR
            net = UNETR(in_channels=self.channels,
                        out_channels=self.num_classes,
                        img_size=tuple(self.input_shape),
                        feature_size=16,
                        hidden_size=768,
                        mlp_dim=3072,
                        num_heads=12,
                        pos_embed='perceptron',
                        norm_name='instance',
                        conv_block=True,
                        res_block=True,
                        dropout_rate=0.0)

        elif net_name == 'unet':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.Unet(encoder_name=self.encoder_name,
                               encoder_weights=None
                               if not self.ex_pre_trained else 'imagenet',
                               in_channels=self.channels,
                               classes=self.num_classes,
                               aux_params={"classes": self.num_classes - 1})
        elif net_name == 'unet++':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.UnetPlusPlus(
                    encoder_name=self.encoder_name,
                    encoder_weights=None
                    if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes,
                    aux_params={"classes": self.num_classes - 1})

        elif net_name == 'deeplabv3+':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.DeepLabV3Plus(
                    encoder_name=self.encoder_name,
                    encoder_weights=None
                    if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes,
                    aux_params={"classes": self.num_classes - 1})

        return net

    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            from loss.cross_entropy import CrossentropyLoss
            loss = CrossentropyLoss(weight=class_weight)

        elif loss_fun == 'TopKLoss':
            from loss.cross_entropy import TopKLoss
            loss = TopKLoss(weight=class_weight, k=self.topk)

        elif loss_fun == 'FocalLoss':
            from loss.cross_entropy import FocalLoss
            loss = FocalLoss(reduction='sum')

        elif loss_fun == 'DiceLoss':
            from loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0, p=1)

        elif loss_fun == 'CEPlusDice':
            from loss.combine_loss import CEPlusDice
            loss = CEPlusDice(weight=class_weight, ignore_index=0)

        elif loss_fun == 'FLPlusDice':
            from loss.combine_loss import FLPlusDice
            loss = FLPlusDice(weight=class_weight, ignore_index=0)

        return loss

    def _get_optimizer(self, optimizer, net, lr):
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)

        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.T_max)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 5, T_mult=2)
        elif lr_scheduler == 'poly_lr':
            lr_scheduler = PolyLR(optimizer,max_epochs=self.n_epoch)
        return lr_scheduler

    def _get_pre_trained(self, weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path, map_location='cpu')
        self.net.load_state_dict(checkpoint['state_dict'])
        if ckpt_point:
            self.start_epoch = checkpoint['epoch'] + 1
            # self.metrics_threshold = eval(os.path.splitext(self.weight_path.split(':')[-2])[0])


# computing tools


class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def binary_dice(predict, target, smooth=1e-5):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[
        0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)  #N，H*W
    target = target.contiguous().view(target.shape[0], -1)  #N，H*W

    inter = torch.sum(torch.mul(predict, target), dim=1)  #N
    union = torch.sum(predict + target, dim=1)  #N

    dice = (2 * inter + smooth) / (union + smooth)  #N

    # nan mean
    # dice_index = dice != 1.0
    # dice = dice[dice_index]

    return dice.mean()


def compute_dice(predict, target, ignore_index=0):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)

    onehot_predict = torch.argmax(predict, dim=1)  #N*H*W
    onehot_target = torch.argmax(target, dim=1)  #N*H*W

    dice_list = np.ones((target.shape[1]), dtype=np.float32)
    for i in range(target.shape[1]):
        if i != ignore_index:
            if i not in onehot_predict and i not in onehot_target:
                continue
            dice = binary_dice((onehot_predict == i).float(),
                               (onehot_target == i).float())
            dice_list[i] = round(dice.item(), 4)
    # dice_list = np.where(dice_list == -1.0, np.nan, dice_list)

    return np.nanmean(dice_list[1:])


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score



class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, ck_epoch=0, exponent=0.9, last_epoch=-1,verbose=False):
        # Initialize some parameters
        self.max_epochs = max_epochs
        self.ck_epoch = ck_epoch
        self.exponent = exponent
        super(PolyLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
                warnings.warn("To get the last learning rate computed by the scheduler, "
                            "please use `get_last_lr()`.", UserWarning)
        # Implement your logic for adjusting the learning rate
        current_epoch = self.last_epoch 

        if (self.last_epoch > self.max_epochs):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [base_lrs  * (1 - (current_epoch - self.ck_epoch) /
                (self.max_epochs - self.ck_epoch)) ** self.exponent for base_lrs in self.base_lrs]

