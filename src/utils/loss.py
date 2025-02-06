import torch
import torch.nn as nn
import torch.nn.functional as F

"""Losses adapted from SAM2 github"""

def focal_loss(preds, gt, nb_objects, alpha = 0.25, gamma = 2.0):
    """Source: https://amaarora.github.io/posts/2020-06-29-FocalLoss.html"""
    proba = torch.sigmoid(preds) # preds should be the logits predicted
    ce_loss = F.binary_cross_entropy_with_logits(preds, gt, reduction = "none")

    p_t = proba * gt + (1 - proba) * (1 - gt)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
    loss = alpha_t * loss

    return loss.mean(1).sum() / nb_objects

def dice_loss(preds, gt, nb_objects):
    preds = torch.sigmoid(preds)

    y_true_flat = gt.view(gt.size(0), -1)
    y_pred_flat = preds.view(preds.size(0), -1)
    intersection = (y_true_flat * y_pred_flat).sum(dim = 1)

    loss = 1 - (2 * intersection + 1) / (y_true_flat.sum(dim = 1) + y_pred_flat.sum(dim = 1) + 1)

    return loss.sum() / nb_objects

def iou_loss(preds, gt, pred_iou, nb_objects, use_l1_loss):
    y_true_flat = (gt.view(gt.size(0), -1) > 0.0).float()
    y_pred_flat = (preds.view(preds.size(0), -1) > 0.0).float()

    intersection = (y_true_flat * y_pred_flat).sum(dim = 1)
    union = y_true_flat.sum(dim = 1) + y_pred_flat.sum(dim = 1) - intersection
    union = torch.clamp(union, min = 1.0)

    ious = intersection / (union + 1e-6)

    if use_l1_loss:
        loss = F.l1_loss(pred_iou, ious, reduction = "none")
    else:
        loss = F.mse_loss(pred_iou, ious, reduction = "none")

    return loss.sum() / nb_objects

class SAM_Loss(nn.Module):
    def __init__(self, focal_weight = 20.0, dice_weight = 1.0, iou_weight = 1.0, alpha = 0.25, gamma = 2.0, is_sam2_loss = False):
        super(SAM_Loss, self).__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.alpha = alpha
        self.gamma = gamma
        self.is_sam2_loss = is_sam2_loss

    def forward(self, best_pred, gt_mask, pred_iou):
        nb_objects = best_pred.size(0) # here one mask per image, thus nb_objects = batch_size

        focal = focal_loss(best_pred, gt_mask, nb_objects = nb_objects,alpha = self.alpha, gamma = self.gamma)
        dice = dice_loss(best_pred, gt_mask, nb_objects)

        if self.is_sam2_loss:
            iou = iou_loss(best_pred, gt_mask, pred_iou, nb_objects, use_l1_loss = True)
        else:
            iou = iou_loss(best_pred, gt_mask, pred_iou, nb_objects, use_l1_loss = False)

        total_loss = self.focal_weight * focal + self.dice_weight * dice + self.iou_weight * iou

        return total_loss, {'focal' : focal.item(), 'dice' : dice.item(), 'iou' : iou.item()}
    
class Custom_SAM2_Loss(nn.Module):
    def __init__(self, score_weight = 0.05):
        super(Custom_SAM2_Loss, self).__init__()
        self.score_weight = score_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, best_pred, gt_mask, pred_iou, threshold):
        if best_pred.ndim == 2:  # single item, no batch dimension
            best_pred = best_pred.unsqueeze(0)
            gt_mask = gt_mask.unsqueeze(0)
            pred_iou = pred_iou.unsqueeze(0)

        seg_loss = self.bce_loss(best_pred, gt_mask).mean()
        binary_pred = torch.where(best_pred > threshold, 1.0, 0.0).float()

        y_true_flat = gt_mask.view(gt_mask.size(0), -1)
        y_pred_flat = binary_pred.view(binary_pred.size(0), -1)

        y_true_sum = y_true_flat.sum(dim = 1)
        y_pred_sum = y_pred_flat.sum(dim = 1)

        intersection = (y_true_flat * y_pred_flat).sum(dim = 1)
        union = y_true_sum + y_pred_sum - intersection

        iou = intersection / (union + 1e-6)
        score_loss = torch.abs(pred_iou - iou).mean()

        total_loss = seg_loss + self.score_weight * score_loss

        return total_loss, {'seg' : seg_loss.item(), 'score' : score_loss.item()}
