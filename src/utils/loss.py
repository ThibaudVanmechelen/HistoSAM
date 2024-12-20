import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(preds, gt, alpha = 0.25, gamma = 2.0):
    """Source: https://amaarora.github.io/posts/2020-06-29-FocalLoss.html"""
    proba = torch.sigmoid(preds) # preds should be the logits predicted
    p_t = torch.where(gt == 1, proba, 1 - proba)

    loss = -alpha * (1 - p_t).pow(gamma) * torch.log(p_t + 1e-8)

    return loss.mean()

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
        y_true_flat = gt_mask.view(gt_mask.size(0), -1)
        y_pred_flat = best_pred.view(best_pred.size(0), -1)

        y_true_sum = y_true_flat.sum(dim = 1)
        y_pred_sum = y_pred_flat.sum(dim = 1)

        intersection = (y_true_flat * y_pred_flat).sum(dim = 1)
        union = y_true_sum + y_pred_sum - intersection

        dice_score = (2 * intersection) / (y_true_sum + y_pred_sum + 1e-6)
        iou_score = intersection / (union + 1e-6)

        focal = focal_loss(best_pred, gt_mask, self.alpha, self.gamma)
        dice = 1 - dice_score.mean()

        if self.is_sam2_loss:
            iou = F.l1_loss(pred_iou.squeeze(), iou_score)
        else:
            iou = F.mse_loss(pred_iou.squeeze(), iou_score) 

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

        return total_loss, {'seg' : seg_loss.item(), 'score' :score_loss.item()}
