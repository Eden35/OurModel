
import os
import torch
import math
import visdom
import torch.utils.data as Data
import argparse
import numpy as np
import sys
from tqdm import tqdm
import random
from thop import profile
from ptflops import get_model_complexity_info

from distutils.version import LooseVersion
from Datasets.CNV import CNV_dataset

from utils.transform import CNV_transform, CNV_transform_320, CNV_transform_newdata
from utils.SetCriterion import SetCriterion
import dataloaders.aggregative_fusion as af
from torch.autograd import Variable
from torchvision.transforms import GaussianBlur
from Models.SAUM_Net import SAUM_Net

from utils.dice_loss import get_soft_label, val_dice_isic,MYIOU,MYHD95,indicators,safe_assd
from utils.dice_loss import Intersection_over_Union_isic
from utils.dice_loss_github import SoftDiceLoss_git, CrossentropyND

from utils.evaluation import AverageMeter
from utils.binary import assd, dc, jc, precision, sensitivity, specificity, F1, ACC,hd
from torch.optim import lr_scheduler

from location_scale_augmentation import FIESTA

from time import *
from PIL import Image
from medpy.metric.binary import hd95


Test_Model = {
    "SAUM_Net": SAUM_Net
}

Test_Dataset = {'CNV': CNV_dataset}

Test_Transform = {'A': CNV_transform, 'B': CNV_transform_320, "C": CNV_transform_newdata}

torch.cuda.empty_cache()


def structure_loss(out_f, target, num_classes=2):
    loss = []
    soft_dice_loss2 = SoftDiceLoss_git(batch_dice=False, dc_log=True)
    CE_loss_F = CrossentropyND()

    for i in range(len(out_f)):
        out_c = out_f[i]
        target_soft_a = get_soft_label(target, num_classes)
        target_soft = target_soft_a.permute(0, 3, 1, 2)
        dice_loss_f = soft_dice_loss2(out_c, target_soft)
        ce_loss_f = CE_loss_F(out_c, target)
        loss_f = dice_loss_f + ce_loss_f
        loss.append(loss_f)

    return sum(loss)


def one_loss(out_c, target, num_classes=2):
    soft_dice_loss2 = SoftDiceLoss_git(batch_dice=False, dc_log=False)
    CE_loss_F = CrossentropyND()

    target_soft_a = get_soft_label(target, num_classes)
    target_soft = target_soft_a.permute(0, 3, 1, 2)
    dice_loss_f = soft_dice_loss2(out_c, target_soft)
    ce_loss_f = CE_loss_F(out_c, target)
    loss_f = dice_loss_f + ce_loss_f

    return loss_f



import torch.nn.functional as F

def dice_loss(pre, target, smooth = 1.):
    pre = F.sigmoid(pre)
    pre = pre.view(-1)
    target = target.view(-1)
    intersection = (pre * target).sum()
    dice = (2. * intersection + smooth)/(pre.sum() + target.sum() + smooth)
    return 1-dice
#二类别
def structure_loss2(pred, mask, type = None):
    mask = mask.to(pred.device)
    mask = mask.repeat(1, 2, 1, 1)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    n, c, _, _ = pred.shape
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    if type == 'bce+iou':
    # allloss = wiou.mean() + wbce.mean()
        allloss = wbce.mean() + wiou.mean()
        return allloss
    elif  type == 'bce':
    # allloss = wiou.mean() + wbce.mean()
        allloss = wbce.mean()
        return allloss
    elif type == 'iou+dice':
        dice = dice_loss(pred, mask)
        allloss =  wiou.mean() + dice.mean()
        return allloss
    else:
        raise Exception



class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

from torch.utils.tensorboard import SummaryWriter

def train(train_loader, model, scheduler, optimizer, args, epoch):

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    mix_losses_meter = AverageMeter()
    mix_losses_meter2 = AverageMeter()

    location_scale = FIESTA(vrange=(0., 1.), background_threshold=0.01)
    model.train()
    for step, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):

        image = x.float().cuda()
        target = y.float().cuda()
        LFAT_pre = location_scale.Local_Location_Scale_Augmentation(image.detach().cpu().numpy(),
                                                                    target.to(torch.int32).detach().cpu().numpy())
        # 不确定性代码
        LFAT_pre = torch.tensor(LFAT_pre)
        LFAT_pre = LFAT_pre.float().cuda()



        optimizer.zero_grad()

        logits, logits2 = model(image, LFAT_pre)

        out_f = torch.sigmoid(logits)
        out_f2 = torch.sigmoid(logits2)
        
        background = out_f[:, 0].unsqueeze(1)  

        
        combined_foreground = torch.clamp(out_f[:, 1] + out_f2[:, 1], min=0, max=1)  # 确保值在 [0, 1] 之间

 
        out_f3 = torch.cat((background, combined_foreground.unsqueeze(1)), dim=1)  # (N, 2, H, W)


        # ---- loss function ----

        if isinstance(out_f, list) or isinstance(out_f, tuple):
            loss1 = structure_loss(out_f, target)
        else:
            loss1 = one_loss(out_f, target)
        losses1.update(loss1.data, image.size(0))
        loss1.backward(retain_graph=True)


        loss2 = structure_loss2(logits2,target, type = 'bce+iou')  #bce+iou  iou+dice
    
        losses2.update(loss2.data, image.size(0))
        loss2.backward(retain_graph=True)


        proba = torch.sigmoid(logits.detach())
        uncertainty = -torch.mean(proba*torch.log2(proba+1e-10), 1)
        uncertainty_1 = af.rescale_intensity(torch.unsqueeze(uncertainty, 1))

        proba2 = torch.sigmoid(logits2.detach())
        uncertainty2 = -torch.mean(proba2 * torch.log2(proba2+1e-10), 1)
        uncertainty_2 = af.rescale_intensity(torch.unsqueeze(uncertainty2, 1))

        mean_UG = (uncertainty_1 + uncertainty_2) / 2
        max_UG = torch.maximum(uncertainty_1, uncertainty_2)
        UG = (mean_UG + max_UG) / 2
        UG = af.rescale_intensity(GaussianBlur(kernel_size=(15, 15), sigma=(5, 5))(UG))

        gt = target.detach().cpu().numpy()
        gt = np.where(gt > 1, 1, gt)
        gt_ug = gt * UG[:, 0].detach().cpu().numpy()

        gt_idx_len = len(np.where(gt == 1)[0])
        uncer_sum = np.sum(gt_ug)


        if gt_idx_len > 0 and (uncer_sum / gt_idx_len) < 1:
            mixed_var = image.detach() * UG + LFAT_pre * (1 - UG)
        else:
            mixed_var = image.detach()+LFAT_pre


        mixed_var = Variable(mixed_var, requires_grad=True)


        mixed_logits,mixed_logits2 = model(mixed_var,mixed_var)
        # 二类别
        mix_out = torch.sigmoid(mixed_logits)
        mix_out2 = torch.sigmoid(mixed_logits2)


        background2 = mix_out[:, 0].unsqueeze(1)  # 变为 (N, 1, H, W)

        combined_foreground2 = torch.clamp(mix_out[:, 1] + mix_out2[:, 1], min=0, max=1)  # 确保值在 [0, 1] 之间

        out_f4 = torch.cat((background2, combined_foreground2.unsqueeze(1)), dim=1)  # (N, 2, H, W)
       
        if isinstance(mix_out, list) or isinstance(mix_out, tuple):
            mix_losses = structure_loss(mix_out, target)
        else:
            mix_losses = one_loss(mix_out, target)

        mix_losses_meter.update(mix_losses.data, image.size(0))

        mix_losses.backward(retain_graph=True)

        mixloss2 = structure_loss2(mixed_logits2,target, type = 'bce+iou')  #bce+iou

        mix_losses_meter2.update(mixloss2.data, image.size(0))
        mixloss2.backward()

        optimizer.step()


        if step % (math.ceil(float(len(train_loader.dataset)) / args.batch_size)) == 0:
            print(
                'current lr: {} | Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {losses.avg:.6f} Loss2: {losses2.avg:.6f} mix_losses: {mix_losses_meter.avg: .6f}'.format(
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    epoch, step * len(image), len(train_loader.dataset),
                           100. * step / len(train_loader), losses=losses1, losses2=losses2, mix_losses_meter=mix_losses_meter))
    # mix_losses: {mix_losses_meter.avg: .6f} ,mix_losses_meter=mix_losses_meter
    print('The average loss:{losses.avg:.4f}'.format(losses=losses1))
    print('The average loss:{mix_losses_meter.avg:.4f}'.format(mix_losses_meter=mix_losses_meter))
    writer.close()

    return losses1.avg



def valid_isic(valid_loader, model, optimizer, args, epoch, best_score, val_acc_log):
    isic_Jaccard = []
    isic_dc = []
    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader), total=len(valid_loader), mininterval=0.001):
        image = t.float().cuda()
        target = k.float().cuda()


        logits, logits2 = model(image, image)
        out_f = torch.sigmoid(logits)
        out_f2 = torch.sigmoid(logits2)

        background = out_f[:, 0].unsqueeze(1)  # 变为 (N, 1, H, W)

        combined_foreground = torch.clamp(out_f[:, 1] + out_f2[:, 1], min=0, max=1)  # 确保值在 [0, 1] 之间

        out_f3 = torch.cat((background, combined_foreground.unsqueeze(1)), dim=1)  # (N, 2, H, W)


        if isinstance(out_f3, list) or isinstance(out_f3, tuple):
            output = out_f3[-1]
        else:
            output = out_f3

        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_Jaccard.append(isic_b_Jaccard)
        isic_dc.append(isic_b_dc)


    isic_Jaccard_mean = np.average(isic_Jaccard)

    isic_dc_mean = np.average(isic_dc)


    net_score = isic_Jaccard_mean + isic_dc_mean

    print('The ISIC Dice score: {dice: .4f}; '
          'The ISIC JC score: {jc: .4f}'.format(
        dice=isic_dc_mean, jc=isic_Jaccard_mean))

    with open(val_acc_log, 'a') as vlog_file:
        line = "{} | {dice: .4f} | {jc: .4f}".format(epoch, dice=isic_dc_mean, jc=isic_Jaccard_mean)
        vlog_file.write(line + '\n')

    if net_score > max(best_score):
        best_score.append(net_score)
        print(best_score)
        modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    return isic_Jaccard_mean, isic_dc_mean, net_score



def Intersection_over_Union_hard(pred, ground, num_class):

    ground = ground.long()

    pred = F.one_hot(pred, num_classes=num_class).float().view(-1, num_class)
    ground = F.one_hot(ground.squeeze(dim=1), num_classes=num_class).float().view(-1, num_class)

    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)

    iou_score = intersect / (ref_vol + seg_vol - intersect + 1e-6)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score



def compute_JAC(output, target, eps=1e-6):
    """
    计算二分类 Jaccard Index (JAC)

    参数:
        output: (N, 2, H, W) - 模型输出，包含背景和前景概率
        target: (N, H, W) - 真实标签，0 表示背景，1 表示前景
        eps: 避免除零的小数值

    返回:
        JAC: float, 二分类的 Jaccard Index
    """
    # 1. 获取预测类别 (N, H, W)
    pred = output

    # 2. 计算整体交集和并集（不按类别分开）
    # intersection = (pred & target).sum().float()  # 交集
    intersection = ((pred > 0.5) & (target > 0.5)).sum().float()

    union = ((pred > 0.5) | (target > 0.5)).sum().float()  # 并集

    # 3. 计算 Jaccard Index
    JAC = (intersection + eps) / (union + eps)  # 避免除零
    return JAC



def test_isic(test_loader, model, args, test_acc_log, date_type, save_img=True):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    isic_dice = []
    isic_iou = []
    isic_assd = []
    isic_acc = []
    isic_sensitive = []
    isic_specificy = []
    isic_precision = []
    isic_f1_score = []
    isic_Jaccard_M = []
    isic_Jaccard_N = []
    isic_Jaccard = []
    isic_JAC = []
    HD = []
    isic_dc = []
    infer_time = []

    print(
        "******************************************************************** {} || start **********************************".format(
            date_type) + "\n")
    # best_score
    modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    img_saved_dir_root = os.path.join(args.ckpt, "segmentation_result")
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    model.eval()
    location_scale = FIESTA(vrange=(0., 1.), background_threshold=0.01)
    for step, (name, img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = img.float().cuda()

        target = lab.float().cuda()  # [batch, 1, 224, 320]
        LFAT_pre = location_scale.Local_Location_Scale_Augmentation(image.detach().cpu().numpy(),
                                                                    target.to(torch.int32).detach().cpu().numpy())
        LFAT_pre = torch.tensor(LFAT_pre)
        LFAT_pre = LFAT_pre.float().cuda()

        begin_time = time()
        logits, logits2 = model(image, image)

        out_f = torch.sigmoid(logits)  #二类别分割
        out_f2 = torch.sigmoid(logits2)

        background = out_f[:, 0].unsqueeze(1)  # 变为 (N, 1, H, W)

        combined_foreground = torch.clamp(out_f[:, 1] + out_f2[:, 1], min=0, max=1)  # 确保值在 [0, 1] 之间

        out_f3 = torch.cat((background, combined_foreground.unsqueeze(1)), dim=1)  # (N, 2, H, W)


        proba = torch.sigmoid(logits.detach())
        uncertainty = -torch.mean(proba * torch.log2(proba + 1e-10), 1)
        uncertainty_1 = af.rescale_intensity(torch.unsqueeze(uncertainty, 1))

        proba2 = torch.sigmoid(logits2.detach())
        uncertainty2 = -torch.mean(proba2 * torch.log2(proba2 + 1e-10), 1)
        uncertainty_2 = af.rescale_intensity(torch.unsqueeze(uncertainty2, 1))

        mean_UG = (uncertainty_1 + uncertainty_2) / 2
        max_UG = torch.maximum(uncertainty_1, uncertainty_2)
        UG = (mean_UG + max_UG) / 2
        UG = af.rescale_intensity(GaussianBlur(kernel_size=(15, 15), sigma=(5, 5))(UG))

        gt = target.detach().cpu().numpy()
        gt = np.where(gt > 1, 1, gt)
        gt_ug = gt * UG[:, 0].detach().cpu().numpy()

        gt_idx_len = len(np.where(gt == 1)[0])
        uncer_sum = np.sum(gt_ug)


        if gt_idx_len > 0 and (uncer_sum / gt_idx_len) < 1.8:
            mixed_var = image.detach() * UG + LFAT_pre * (1 - UG)
        else:

            mixed_var = image.detach()+LFAT_pre

        LFAT_pre1 = location_scale.Local_Location_Scale_Augmentation(image.detach().cpu().numpy(),
                                                                    UG.to(torch.int32).detach().cpu().numpy())

        LFAT_pre1 = torch.tensor(LFAT_pre1)
        LFAT_pre1 = LFAT_pre1.float().cuda()

        logits4,logits5 = model(image,LFAT_pre1)

        out_f4 = torch.sigmoid(logits4)  #二类别分割
        out_f5 = torch.sigmoid(logits5)

        background = out_f4[:, 0].unsqueeze(1)  # 变为 (N, 1, H, W)

        combined_foreground = torch.clamp(out_f4[:, 1] + out_f5[:, 1], min=0, max=1)  # 确保值在 [0, 1] 之间

        out_f6 = torch.cat((background, combined_foreground.unsqueeze(1)), dim=1)  # (N, 2, H, W)


        if isinstance(out_f6, list) or isinstance(out_f6, tuple):
            output = out_f6[-1]
        else:
            output = out_f6


        if isinstance(out_f4, list) or isinstance(out_f4, tuple):
            output4 = out_f4[-1]
        else:
            output4 = out_f4
        if isinstance(out_f5, list) or isinstance(out_f5, tuple):
            output5 = out_f5[-1]
        else:
            output5 = out_f5



        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)
        #二类别
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)

        output_dis11 = torch.max(output4, 1)[1].unsqueeze(dim=1)
        output_dis22 = torch.max(output5, 1)[1].unsqueeze(dim=1)



        output_dis1 = torch.max(output, 1)[1]

        """
        save segmentation result
        """
        if save_img:
            if date_type == "CNV" and args.val_folder == "folder1":
                npy_path = os.path.join(args.root_path, 'image', name[0])
                img = np.load(npy_path)
                im = Image.fromarray(np.uint8(img))
                im_path = name[0].split(".")[0] + "_img" + ".png"
                img_saved_dir = os.path.join(img_saved_dir_root, name[0].split(".")[0])
                if not os.path.isdir(img_saved_dir):
                    os.makedirs(img_saved_dir)
                im.save(os.path.join(img_saved_dir, im_path))

                target_np = target.squeeze().cpu().numpy()
                label = Image.fromarray(np.uint8(target_np * 255))
                label_path = name[0].split(".")[0] + "_label" + ".png"
                label.save(os.path.join(img_saved_dir, label_path))

                seg_np = output_dis.squeeze().cpu().numpy()
                seg = Image.fromarray(np.uint8(seg_np * 255))
                seg_path = name[0].split(".")[0] + "_seg" + ".png"
                seg.save(os.path.join(img_saved_dir, seg_path))


                seg_np1 = output_dis11.squeeze().cpu().numpy()
                seg1 = Image.fromarray(np.uint8(seg_np1 * 255))
                seg1_path = name[0].split(".")[0] + "_seg1" + ".png"
                seg1.save(os.path.join(img_saved_dir, seg1_path))

                seg_np2 = output_dis22.squeeze().cpu().numpy()
                seg2 = Image.fromarray(np.uint8(seg_np2 * 255))
                seg2_path = name[0].split(".")[0] + "_seg2" + ".png"
                seg2.save(os.path.join(img_saved_dir, seg2_path))


                LFAT_np = LFAT_pre.squeeze(0)
                LFAT_np = LFAT_np.detach().cpu().numpy()
                LFAT_np = np.transpose(LFAT_np, (1, 2, 0))
                LFAT = Image.fromarray(np.uint8(LFAT_np * 255))
                LFAT_path = name[0].split(".")[0] + "_LFAT" + ".png"
                LFAT.save(os.path.join(img_saved_dir, LFAT_path))

                UG1_np = (1-UG).squeeze().detach().cpu().numpy()
                UG1 = Image.fromarray(np.uint8(UG1_np * 255))
                UG1_path = name[0].split(".")[0] + "_1UG" + ".png"
                UG1.save(os.path.join(img_saved_dir, UG1_path))

                UG_np = UG.squeeze().detach().cpu().numpy()
                UG = Image.fromarray(np.uint8(UG_np * 255))
                UG_path = name[0].split(".")[0] + "_UG" + ".png"
                UG.save(os.path.join(img_saved_dir, UG_path))

                mixed_var_np = mixed_var.squeeze().detach().cpu().numpy()  # 移除多余的维度
                mixed_var_np = mixed_var_np.transpose(1, 2, 0)  # 调整为 (224, 224, 3)
                mixed = Image.fromarray(np.uint8(mixed_var_np * 255))
                mixed_path = name[0].split(".")[0] + "_mixed" + ".png"
                mixed.save(os.path.join(img_saved_dir, mixed_path))

            else:
                pass
        else:
            pass

        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        output_soft = get_soft_label(output_dis, 2)
        target_soft = get_soft_label(target, 2)

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)


        isic_b_dice = val_dice_isic(output_soft, target_soft, 2)  # the dice
        isic_b_iou = Intersection_over_Union_hard(output_dis,target,2)
        isic_b_jac = compute_JAC(output_dis1, target)
        isic_b_HD = MYHD95(output_dis_test,target_test,2)
        isic_b_asd = safe_assd(output_arr[:, :, 1], label_arr[:, :, 1])
        isic_b_acc = ACC(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the accuracy
        isic_b_sensitive = sensitivity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the sensitivity
        isic_b_specificy = specificity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the specificity
        isic_b_precision = precision(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the precision
        isic_b_f1_score = F1(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the F1
        isic_b_Jaccard_m = jc(output_arr[:, :, 1], label_arr[:, :, 1])  # the Jaccard melanoma
        isic_b_Jaccard_n = jc(output_arr[:, :, 0], label_arr[:, :, 0])  # the Jaccard no-melanoma
        isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())

        dice_np = isic_b_dice.data.cpu().numpy()
        iou_np = isic_b_iou.data.cpu().numpy()
        JAC = isic_b_jac.detach().cpu().numpy()
        b_HD = isic_b_HD if isinstance(isic_b_HD, float) else isic_b_HD.detach().cpu().numpy()

        isic_dice.append(dice_np)
        isic_iou.append(iou_np)
        isic_JAC.append(JAC)
        HD.append(b_HD)
        isic_assd.append(isic_b_asd)
        isic_acc.append(isic_b_acc)
        isic_sensitive.append(isic_b_sensitive)
        isic_specificy.append(isic_b_specificy)
        isic_precision.append(isic_b_precision)
        isic_f1_score.append(isic_b_f1_score)
        isic_Jaccard_M.append(isic_b_Jaccard_m)
        isic_Jaccard_N.append(isic_b_Jaccard_n)
        isic_Jaccard.append(isic_b_Jaccard)
        isic_dc.append(isic_b_dc)

        if date_type == "CNV":
            with open(test_acc_log, 'a') as tlog_file:
                line = "{} | {dice: .4f} | {jc: .4f}".format(name[0], dice=isic_b_dc, jc=isic_b_Jaccard)
                tlog_file.write(line + '\n')
        else:
            print("can not supports dataset: {}", date_type)


    all_time = np.sum(infer_time)
    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)
    isic_JAC_mean = np.average(isic_JAC)
    isic_JAC_std = np.std(isic_JAC)

    isic_HD_mean = np.average(HD)
    isic_HD_std = np.std(HD)
    isic_assd_mean = np.average(isic_assd)
    isic_assd_std = np.std(isic_assd)

    isic_acc_mean = np.average(isic_acc)
    isic_acc_std = np.std(isic_acc)

    isic_sensitive_mean = np.average(isic_sensitive)
    isic_sensitive_std = np.std(isic_sensitive)

    isic_specificy_mean = np.average(isic_specificy)
    isic_specificy_std = np.std(isic_specificy)

    isic_precision_mean = np.average(isic_precision)
    isic_precision_std = np.std(isic_precision)

    isic_f1_score_mean = np.average(isic_f1_score)
    iisic_f1_score_std = np.std(isic_f1_score)

    isic_Jaccard_M_mean = np.average(isic_Jaccard_M)
    isic_Jaccard_M_std = np.std(isic_Jaccard_M)

    isic_Jaccard_N_mean = np.average(isic_Jaccard_N)
    isic_Jaccard_N_std = np.std(isic_Jaccard_N)

    isic_Jaccard_mean = np.average(isic_Jaccard)
    isic_Jaccard_std = np.std(isic_Jaccard)

    isic_dc_mean = np.average(isic_dc)
    isic_dc_std = np.std(isic_dc)


    print('The mean dice: {isic_dice_mean: .4f}; The dice std: {isic_dice_std: .4f}'.format(
        isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std))
    print('The mean IoU: {isic_iou_mean: .4f}; The IoU std: {isic_iou_std: .4f}'.format(
        isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std))

    print('The mean JAC: {isic_JAC_mean: .4f}; The JAC std: {isic_JAC_std: .4f}'.format(
        isic_JAC_mean=isic_JAC_mean, isic_JAC_std=isic_JAC_std))

    print('The mean HD: {isic_HD_mean: .4f}; The HD std: {isic_HD_std: .4f}'.format(
        isic_HD_mean=isic_HD_mean, isic_HD_std=isic_HD_std))
    print('The mean assd: {isic_assd_mean: .4f}; The assd std: {isic_assd_std: .4f}'.format(
        isic_assd_mean=isic_assd_mean, isic_assd_std=isic_assd_std))

    print('The mean ACC: {isic_acc_mean: .4f}; The ACC std: {isic_acc_std: .4f}'.format(
        isic_acc_mean=isic_acc_mean, isic_acc_std=isic_acc_std))
    print('The mean sensitive: {isic_sensitive_mean: .4f}; The sensitive std: {isic_sensitive_std: .4f}'.format(
        isic_sensitive_mean=isic_sensitive_mean, isic_sensitive_std=isic_sensitive_std))
    print('The mean specificy: {isic_specificy_mean: .4f}; The specificy std: {isic_specificy_std: .4f}'.format(
        isic_specificy_mean=isic_specificy_mean, isic_specificy_std=isic_specificy_std))
    print('The mean precision: {isic_precision_mean: .4f}; The precision std: {isic_precision_std: .4f}'.format(
        isic_precision_mean=isic_precision_mean, isic_precision_std=isic_precision_std))
    print('The mean f1_score: {isic_f1_score_mean: .4f}; The f1_score std: {iisic_f1_score_std: .4f}'.format(
        isic_f1_score_mean=isic_f1_score_mean, iisic_f1_score_std=iisic_f1_score_std))
    print('The mean Jaccard_M: {isic_Jaccard_M_mean: .4f}; The Jaccard_M std: {isic_Jaccard_M_std: .4f}'.format(
        isic_Jaccard_M_mean=isic_Jaccard_M_mean, isic_Jaccard_M_std=isic_Jaccard_M_std))
    print('The mean Jaccard_N: {isic_Jaccard_N_mean: .4f}; The Jaccard_N std: {isic_Jaccard_N_std: .4f}'.format(
        isic_Jaccard_N_mean=isic_Jaccard_N_mean, isic_Jaccard_N_std=isic_Jaccard_N_std))
    print('The mean Jaccard: {isic_Jaccard_mean: .4f}; The Jaccard std: {isic_Jaccard_std: .4f}'.format(
        isic_Jaccard_mean=isic_Jaccard_mean, isic_Jaccard_std=isic_Jaccard_std))
    print('The mean dc: {isic_dc_mean: .4f}; The dc std: {isic_dc_std: .4f}'.format(
        isic_dc_mean=isic_dc_mean, isic_dc_std=isic_dc_std))
    print('The inference time: {time: .4f}'.format(time=all_time))

    print(
        "******************************************************************** {} || end **********************************".format(
            date_type) + "\n")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args, val_acc_log, test_acc_log):
    best_score = [0]
    start_epoch = args.start_epoch
    print('loading the {0},{1},{2} dataset ...'.format('train', 'validation', 'test'))
    trainset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='train',
                                       with_name=False, transform=Test_Transform[args.transform])
    validset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='validation',
                                       with_name=False, transform=Test_Transform[args.transform])
    testset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                      with_name=True, transform=Test_Transform[args.transform])

    trainloader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=6)
    validloader = Data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
    testloader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)



    print('Loading is done\n')

    args.num_input
    args.num_classes
    args.out_size
    print("args.out_size: ", args.out_size)
    print("args.h_init_type is: ", args.h_init_type)

    model = Test_Model[args.id](classes=2, channels=3)
    model = model.cuda()
    model1 = Test_Model[args.id](classes=2, channels=3)
    model1 = model1.cuda()


    input = torch.randn(1, 3, args.out_size[0], args.out_size[1])  # batch_size = 1
    LFAT_pre = torch.randn(1, 3, args.out_size[0], args.out_size[1])  # batch_size = 1

    flops, params = profile(model1, inputs=(input.cuda(),LFAT_pre.cuda()))
    print(
        "---------------------------------------------------------------------------------------------------------------------")
    print("\n")
    print("profile test result: ")
    print("Flops: {} G".format(flops / 1e9))
    print("params: {} M".format(params / 1e6))
    input = input.cuda()
    LFAT_pre = LFAT_pre.cuda()
    for _ in range(10):
        _ = model1(input,LFAT_pre)

    # 测量时间
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    n_test = 100
    for _ in range(n_test):
        _ = model1(input,LFAT_pre)
    torch.cuda.synchronize()
    end_time = time.time()

    fps = n_test / (end_time - start_time)
    print("FPS: {:.2f}".format(fps))


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=args.weight_decay)


    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)  # lr_3
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    print("Start training ...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        scheduler.step()
        train_avg_loss = train(trainloader, model, scheduler, optimizer, args, epoch)
        isic_Jaccard_mean, isic_dc_mean, net_score = valid_isic(validloader, model, optimizer, args, epoch, best_score,
                                                                val_acc_log)

        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
                print('the model will be saved at {}'.format(filename))
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                torch.save(state, filename)

    print('Training Done! Start testing')

    test_isic(testloader, model, args, test_acc_log, "CNV", save_img=args.save_img)

    print('Testing Done!')


if __name__ == '__main__':



    # setup_seed(200)
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # gpu-id

    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'
    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')

    parser.add_argument('--id', default="SAUM_Net",
                        help='SAUM_Net')  # Select a loaded model name

    # Path related arguments
    parser.add_argument('--root_path', default='/home/data/project2/demo2/data/CNV',
                        help='root directory of training data')
    parser.add_argument('--ckpt', default='./data/saved_models/',
                        help='folder to output checkpoints')  # The folder in which the trained model is saved
    parser.add_argument('--transform', default='C', type=str,
                        help='which CNV_transform to choose')
    parser.add_argument('--h_init_type', default='m_0',
                        type=str)  # m_0--> zero; m_1--> torch.randn(); m_2--> image_resize;  m_3--> m_1+m_2

    parser.add_argument('--data', default='CNV', help='choose the dataset')
    parser.add_argument('--out_size', default=(224, 224), help='the output image size')
    parser.add_argument('--val_folder', default='folder1', type=str,
                        help='folder1、folder2、folder3')  # five-fold cross-validation

    parser.add_argument('--seed', type=int, default=1234, help='random seed')  # default 1234
    parser.add_argument('--save_img', type=str, default=True, help='whether save segmentation result')


    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 12)')  # batch_size
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')  # default=1e-4
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=400, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')
    args = parser.parse_args()

    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    logfile = os.path.join(args.ckpt, '{}_{}.txt'.format(args.val_folder, args.id))  # path of the training log
    sys.stdout = Logger(logfile)

    val_acc_log = os.path.join(args.ckpt, 'val_acc_{}_{}.txt'.format(args.val_folder, args.id))
    test_acc_log = os.path.join(args.ckpt, 'test_acc_{}_{}.txt'.format(args.val_folder, args.id))

    print('Models are saved at %s' % (args.ckpt))
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args, val_acc_log, test_acc_log)
