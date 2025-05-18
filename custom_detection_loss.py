import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.loss import v8DetectionLoss, FocalLoss, VarifocalLoss

class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.custom_identifier = "CUSTOM_VarifocalLoss_LOSS_v1"
        print(f"\n=== {self.custom_identifier} Initialized ===")
        print("Using VarifocalLoss for classification")
        self.VarifocalLoss = VarifocalLoss()
        
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
        
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
        
    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        print(f"\nUsing {self.custom_identifier} for loss calculation")
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # Handle empty batch
        if batch["batch_idx"].shape[0] == 0:
            return loss.sum() * batch_size, loss.detach()
        
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # # Create one-hot encoded target labels for VarifocalLoss loss
        target_labels_onehot = torch.zeros(
             (target_labels.shape[0], target_labels.shape[1], self.nc),
             dtype=torch.int64,
             device=target_labels.device,
        )
        target_labels_onehot.scatter_(2, target_labels.unsqueeze(-1).long(), 1)
        
        # Calculate number of ground truth objects
        num_gts = fg_mask.sum()
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Cls loss using VarifocalLoss loss
      
        loss[1] = self.VarifocalLoss(pred_scores, target_labels_onehot.float())
        loss[1] /= num_gts
        #print("raw VarifocalLoss :",loss[1])
        #loss[1] /= target_scores_sum  # Normalize by number of ground truth objects
        
            
        # Bbox loss (using parent class implementation)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            
        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp['dfl']  # dfl gain
        
        # Print loss values for monitoring
        print("\nLoss Components:")
        print(f"Box Loss: {loss[0].item():.4f}")
        print(f"Classification Loss: {loss[1].item():.4f}")
        print(f"DFL Loss: {loss[2].item():.4f}")
        print(f"Total Loss: {loss.sum().item():.4f}")
        
        return loss.sum() * batch_size, loss.detach() 
