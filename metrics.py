import numpy as np
from sklearn.metrics import confusion_matrix
import copy
import SimpleITK as sitk

class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=0):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        if len(ground_truth.shape) > 1 or len(prediction.shape) > 1:
            ground_truth = ground_truth.flatten()
            prediction = prediction.flatten()
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_mIoU(self,smooth=1e-5):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        intersection_over_union = (intersection + smooth ) / (union.astype(np.float32) + smooth)
        iou_list = [round(case,4) for case in intersection_over_union]
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union, iou_list
    
    def init_op(self):
        self.overall_confusion_matrix = None





class RunningDice():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Dice 
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=0):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        if len(ground_truth.shape) > 1 or len(prediction.shape) > 1:
            ground_truth = ground_truth.flatten()
            prediction = prediction.flatten()
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_dice(self,smooth=1e-5):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set

        intersection_over_union = (2*intersection + smooth ) / (union.astype(np.float32) + smooth)
        dice_list = [round(case,4) for case in intersection_over_union]
        mean_intersection_over_union = np.mean(intersection_over_union[1:])
        
        return mean_intersection_over_union, dice_list
    
    def init_op(self):
        self.overall_confusion_matrix = None




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


def multi_dice(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    dice_list = []
    for i in range(num_classes):
        dice = cal_score(predict==i+1,target==i+1)['Dice']
        dice_list.append(dice)
    
    dice_list = [round(case, 4) for case in dice_list]
    
    return dice_list, round(np.mean(dice_list),4)



def multi_hd(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    hd_list = []
    for i in range(num_classes):
        hd = cal_score(predict==i+1,target==i+1)['HausdorffDistance95']
        hd_list.append(hd)
    
    hd_list = [round(case, 4) for case in hd_list]
    
    return hd_list, round(np.mean(hd_list),4)



def multi_vs(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    vs_list = []
    for i in range(num_classes):
        vs = cal_score(predict==i+1,target==i+1)['VolumeSimilarity']
        vs_list.append(vs)
    
    vs_list = [round(case, 4) for case in vs_list]
    
    return vs_list, round(np.mean(vs_list),4)



def multi_jc(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    jc_list = []
    for i in range(num_classes):
        jc = cal_score(predict==i+1,target==i+1)['Jaccard']
        jc_list.append(jc)
    
    jc_list = [round(case, 4) for case in jc_list]
    
    return jc_list, round(np.mean(jc_list),4)