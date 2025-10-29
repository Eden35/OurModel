import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from scipy.ndimage import binary_erosion
from utils.binary import assd
class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=3, weight_map=None, eps=1e-8):
        dice_loss_ave, dice_score_lesion = soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map)
        return dice_loss_ave, dice_score_lesion

def MYIOU(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    # pred = prediction.contiguous().view(-1, num_class)
    pred = prediction.view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    # dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1e-6)
    iou = dice_score/(2-dice_score)
    iou_mean_score = torch.mean(iou)

    return iou_mean_score

def extract_boundary(mask):
    """
    提取二值 mask 的边界
    """
    eroded = np.zeros_like(mask)
    eroded[1:-1,1:-1] = mask[1:-1,1:-1] & mask[:-2,1:-1] & mask[2:,1:-1] & mask[1:-1,:-2] & mask[1:-1,2:]
    boundary = mask ^ eroded
    return boundary

from medpy.metric.binary import hd95
def MYHD95(prediction, soft_ground_truth, num_class, voxelspacing=None):
    """
    Compute mean HD95 between prediction and ground truth.
    prediction: torch.Tensor (N, C, H, W)  # model outputs (logits or probabilities)
    soft_ground_truth: torch.Tensor (N, C, H, W)  # one-hot or soft labels
    num_class: int
    voxelspacing: spacing for HD calculation, default None
    """
    prediction = prediction.detach().cpu()
    soft_ground_truth = soft_ground_truth.detach().cpu()


    pred_mask=prediction.numpy()
    gt_mask=soft_ground_truth.numpy()
    hd95_list = []

    for n in range(pred_mask.shape[0]):  # loop over batch
        for c in range(1, num_class):  # skip background=0
            pred_bin = (pred_mask[n] == c).astype(np.uint8)
            gt_bin = (gt_mask[n] == c).astype(np.uint8)

            if pred_bin.sum() == 0 and gt_bin.sum() == 0:
                hd95_val = 0.0  # both empty -> perfect
            elif pred_bin.sum() == 0 or gt_bin.sum() == 0:
                hd95_val = np.nan  # or set to large value (e.g., 100)
            else:
                try:
                    hd95_val = hd95(pred_bin, gt_bin, voxelspacing=voxelspacing)
                except RuntimeError:
                    hd95_val = np.nan

            hd95_list.append(hd95_val)


    if len(hd95_list) == 0 or np.all(np.isnan(hd95_list)):
        hd95_mean = 100  # 或者其他合适的默认值
    else:
        hd95_mean = np.nanmean(hd95_list)

    return torch.tensor(hd95_mean, dtype=torch.float32)

def safe_assd(pred, gt, default_value=100.0):
    """
    计算 ASSD，自动处理全 0 输入
    pred, gt: torch.Tensor or np.ndarray, 二值 mask
    default_value: 当预测或标签没有前景时返回的默认值
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()

    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    # 如果预测或标签没有前景，返回默认值
    if pred.sum() == 0 or gt.sum() == 0:
        return torch.tensor(default_value, dtype=torch.float32)

    return torch.tensor(assd(pred, gt), dtype=torch.float32)



from medpy.metric.binary import hd95
def indicators(output, target):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5


    hd95_ = hd95(output_, target_)

    return hd95_



def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
    # dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss
    dice_loss = -torch.log(dice_score)
    dice_loss_ave = torch.mean(dice_loss)
    dice_score_lesion = dice_loss[1]
    return dice_loss_ave, dice_score_lesion

def IOU_loss(prediction, soft_ground_truth, num_class):
    predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_loss = torch.mean(-torch.log(iou_score))

    return iou_loss

def jc_loss(prediction, soft_ground_truth, num_class):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict[:,:,:,1].contiguous().view(-1, num_class)
   # pred = prediction[:,:,:,1].view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth[:,:,:,1].view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    #jc = 10*(1-iou_score)
    jc = 20*torch.mean(-torch.log(iou_score))

    return jc


def val_dice_fetus(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score)
    placenta_dice = dice_score[1]
    brain_dice = dice_score[2]

    return placenta_dice, brain_dice


def Intersection_over_Union_fetus(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    dice_mean_score = torch.mean(iou_score)
    placenta_iou = iou_score[1]
    brain_iou = iou_score[2]

    return placenta_iou, brain_iou


def val_dice_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
   # pred = prediction.contiguous().view(-1, num_class)
    pred = prediction.view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
   # dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1e-6)
    dice_mean_score = torch.mean(dice_score)

    return dice_mean_score
    

def val_dice_isic_v1(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
   # pred = prediction.contiguous().view(-1, num_class)
    pred = prediction.view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
   # dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    smooth = 0.001
    dice_score = 2.0 * (intersect+smooth) / (ref_vol + seg_vol + smooth)
    dice_mean_score = torch.mean(dice_score)

    return dice_mean_score
    


def val_dice_isic_raw0(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
   # dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1e-6)
    dice_mean_score = torch.mean(dice_score)

    return dice_mean_score


def Intersection_over_Union_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    # iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1e-6)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score

def Intersection_over_Union_isic_v1(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    smooth = 0.001
    iou_score = (intersect+smooth )/ (ref_vol + seg_vol - intersect + smooth)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score
