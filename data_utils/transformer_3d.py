import random
from skimage.transform import warp
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
import numpy as np

class RandomCrop3D(object):
  def __init__(self, shape):
    self.shape = shape
    assert len(self.shape) == 3, 'shape error'

  def __call__(self, sample):
    ct = sample['ct']
    seg = sample['seg']
    
    for i in range(len(self.shape)):
        if ct.shape[i+1] > self.shape[i]:
            b = random.randint(0,ct.shape[i+1] - self.shape[i])
            if i == 0:
                ct = ct[:,b:b+self.shape[i],:,:]
                seg = seg[b:b+self.shape[i],:,:]
            if i == 1:
                ct = ct[:,:,b:b+self.shape[i],:]
                seg = seg[:,b:b+self.shape[i],:]
            if i == 2:
                ct = ct[:,:,:,b:b+self.shape[i]]
                seg = seg[:,:,b:b+self.shape[i]]

    new_sample = { 'ct': ct, 'seg':seg}

    return new_sample


class RandomTranslationRotationZoom3D(object):
    '''
    Data augmentation method.
    Including random translation, rotation and zoom, which keep the shape of input.
    Args:
    - mode: string, consisting of 't','r' or 'z'. Optional methods and 'trz'is default.
            't'-> translation,
            'r'-> rotation,
            'z'-> zoom.
    '''
    def __init__(self, mode='trz',num_class=2):
        self.mode = mode
        self.num_class = num_class

    def __call__(self, sample):
        ct = sample['ct']
        seg = sample['seg']
        # get transform coordinate
        img_size = seg.shape
        coords0, coords1, coords2 = np.mgrid[:img_size[0], :img_size[1], :
                                             img_size[2]]
        coords = np.array([
            coords0 - img_size[0] / 2, coords1 - img_size[1] / 2,
            coords2 - img_size[2] / 2
        ])
        tform_coords = np.append(coords.reshape(3, -1),
                                 np.ones((1, np.prod(img_size))),
                                 axis=0)
        # transform configuration
        # translation
        if 't' in self.mode:
            translation = [
                0, np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            ]
        else:
            translation = [0, 0, 0]

        # rotation
        if 'r' in self.mode:
            rotation = euler2mat(
                np.random.uniform(-5, 5) / 180.0 * np.pi, 0, 0, 'sxyz')
        else:
            rotation = euler2mat(0, 0, 0, 'sxyz')

        # zoom
        if 'z' in self.mode:
            zoom = [
                1, np.random.uniform(0.9, 1.1),
                np.random.uniform(0.9, 1.1)
            ]
        else:
            zoom = [1, 1, 1]

        # compose
        warp_mat = compose(translation, rotation, zoom)

        # transform
        w = np.dot(warp_mat, tform_coords)
        w[0] = w[0] + img_size[0] / 2
        w[1] = w[1] + img_size[1] / 2
        w[2] = w[2] + img_size[2] / 2
        warp_coords = w[0:3].reshape(3, img_size[0], img_size[1], img_size[2])

        for i in range(ct.shape[0]):
            ct[i] = warp(ct[i], warp_coords)
        new_label = np.zeros(seg.shape, dtype=np.float32)
        for z in range(1,self.num_class):
            temp = warp((seg == z).astype(np.float32),warp_coords)
            new_label[temp >= 0.5] = z
        seg = new_label   
        new_sample = {'ct': ct, 'seg':seg}

        return new_sample


class RandomFlip3D(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.

    '''
    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        # image: numpy array, (D,H,W)
        # label: integer, 0,1,..
        ct = sample['ct']
        seg = sample['seg']

        if 'h' in self.mode and 'v' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                ct = ct[:,:, ::-1, ...]
                seg = seg[:, ::-1, ...]
            else:
                ct = ct[..., ::-1]
                seg = seg[..., ::-1]

        elif 'h' in self.mode:
            ct = ct[:,:, ::-1, ...]
            seg = seg[:, ::-1, ...]

        elif 'v' in self.mode:
            ct = ct[..., ::-1]
            seg = seg[..., ::-1]
        # avoid the discontinuity of array memory
        ct = ct.copy()
        seg = seg.copy()
        new_sample = {'ct': ct, 'seg':seg}

        return new_sample
