import os
import h5py
import numpy as np
import torch
import SimpleITK as sitk



def cal_score(predict,target):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    overlap_measures_filter.Execute(target, predict)
    Jaccard = overlap_measures_filter.GetJaccardCoefficient()
    Dice = overlap_measures_filter.GetDiceCoefficient()
    VolumeSimilarity = overlap_measures_filter.GetVolumeSimilarity()
    FalseNegativeError = overlap_measures_filter.GetFalseNegativeError()
    FalsePositiveError = overlap_measures_filter.GetFalsePositiveError()
    # print(Jaccard,Dice,VolumeSimilarity,FalseNegativeError,FalsePositiveError)
    
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    
    try:
        hausdorff_distance_filter.Execute(target, predict)
    except RuntimeError:
        result = {
            'Jaccard':Jaccard,
            'Dice':Dice,
            'VolumeSimilarity':VolumeSimilarity,
            'FalseNegativeError':FalseNegativeError,
            'FalsePositiveError':FalsePositiveError,
            'HausdorffDistance':np.nan,
            'HausdorffDistance95':np.nan
        }
        return result
    HausdorffDistance = hausdorff_distance_filter.GetHausdorffDistance()
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(predict, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(predict)

    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside 
    # relationship, is irrelevant)
    # label = 1
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(target, squaredDistance=False))
    reference_surface = sitk.LabelContour(target)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In 
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two 
    # segmentations, though in our case it is. More on this below.
    # mean_surface_distance = np.mean(all_surface_distances)
    # median_surface_distance = np.median(all_surface_distances)
    # std_surface_distance = np.std(all_surface_distances)
    # max_surface_distance = np.max(all_surface_distances)
    HausdorffDistance95 = np.percentile(all_surface_distances,95)
    # print(hd_95)
    # print(HausdorffDistance)
    result = {
        'Jaccard':Jaccard,
        'Dice':Dice,
        'VolumeSimilarity':VolumeSimilarity,
        'FalseNegativeError':FalseNegativeError,
        'FalsePositiveError':FalsePositiveError,
        'HausdorffDistance':HausdorffDistance,
        'HausdorffDistance95':HausdorffDistance95
    }
    return result

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


