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
    def __init__(self, focal_weight = 20.0, dice_weight = 1.0, iou_weight = 1.0, alpha = 0.25, gamma = 2.0):
        super(SAM_Loss, self).__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, best_pred, gt_masks, pred_iou):
        y_true_flat = gt_masks.view(gt_masks.size(0), -1)
        y_pred_flat = best_pred.view(best_pred.size(0), -1)

        y_true_sum = y_true_flat.sum(dim = 1)
        y_pred_sum = y_pred_flat.sum(dim = 1)

        intersection = (y_true_flat * y_pred_flat).sum(dim = 1)
        union = y_true_sum + y_pred_sum - intersection

        dice_score = (2 * intersection) / (y_true_sum + y_pred_sum + 1e-6)
        iou_score = intersection / (union + 1e-6)

        focal = focal_loss(best_pred, gt_masks, self.alpha, self.gamma)
        dice = 1 - dice_score.mean()
        iou = F.mse_loss(pred_iou.squeeze(), iou_score) 

        total_loss = self.focal_weight * focal + self.dice_weight * dice + self.iou_weight * iou

        return total_loss, {'focal' : focal.item(), 'dice' : dice.item(), 'iou' : iou.item()}