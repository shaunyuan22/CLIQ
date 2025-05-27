import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, Linear
from mmcv.runner import BaseModule, auto_fp16
from ..builder import ROTATED_NECKS, build_loss
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.anchor.point_generator import MlvlPointGenerator

@ROTATED_NECKS.register_module()
class ICQ(BaseModule):

    def __init__(self,
                 threshold=0.7,
                 cls_channels=8,
                 feat_channels=256,
                 gamma=2,
                 tau=2.0,
                 alpha=20.0,
                 n=0.5,
                ):
        super(ICQ, self).__init__()

        self.threshold = threshold
        self.cls_out_channels = cls_channels
        self.feat_channels = feat_channels
        self.gamma = gamma
        self.tau = tau
        self.n = n

        point_stride = [4]
        self.prior_generator = MlvlPointGenerator(point_stride)
        self.alpha = alpha

        self._init_layers()

    def _init_layers(self):
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.feat_conv1 = ConvModule(self.feat_channels,
                                    self.feat_channels,
                                    3,
                                    padding=1)

        self.out_conv = self.conv_cen = nn.Conv2d(self.feat_channels, 
                                                  self.cls_out_channels, 3, padding=1)

    def forward(self,
                x,
                **kwargs):
        feat = x[0]
        cen_score = self.out_conv(self.up(
                                    self.feat_conv1(feat)
                                    ))

        return cen_score
        
    def forward_train(self,
                      x, 
                      gt_bboxes, gt_labels,
                      img_metas=None,
                      **kwargs):
        center_score = self(x)
        query_loss = self.loss(center_score, gt_bboxes, 
                               gt_labels, img_metas)
        return query_loss
    
    def loss(self, center_score, imgs_gt_bboxes, 
                   gt_labels, img_metas):
        f_size = [(int(img_metas[0]['ori_shape'][0]/4), int(img_metas[0]['ori_shape'][1]/4))]
        points = self.prior_generator.grid_priors(
                    f_size,
                    dtype=center_score.dtype,
                    device=center_score.device)[0]

        losses_query = torch.zeros(1, device=center_score.device)
        for img_idx in range(len(img_metas)):
            if imgs_gt_bboxes[img_idx].numel() == 0:
                continue
            gt_bboxes = imgs_gt_bboxes[img_idx].unsqueeze(0)
            gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
            cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
            rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                                    dim=-1).reshape(1, -1, 2, 2)
            offset = points.unsqueeze(1) - gt_ctr 
            rotated_offset = torch.matmul(rot_matrix, offset[..., None])
            rotated_offset = rotated_offset.squeeze(-1)

            w, h = gt_wh[..., 0], gt_wh[..., 1]
            offset_x, offset_y = rotated_offset[..., 0], rotated_offset[..., 1]
            left = w / 2 + offset_x
            right = w / 2 - offset_x
            top = h / 2 + offset_y
            bottom = h / 2 - offset_y
            GC_x = torch.abs(torch.min(left, right)/torch.max(left, right))
            GC_y = torch.abs(torch.min(top, bottom)/torch.max(top, bottom))
            GC_xy = torch.pow(GC_x, self.n) * torch.pow(GC_y, self.n)
            inner_allgts_mask = (abs(rotated_offset) < (gt_wh/2)).all(dim=-1)
            GC_target = torch.max(GC_xy * inner_allgts_mask, dim=1)[0]

            center_score_list = center_score[img_idx].permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            center_score_list = torch.max(center_score_list, dim=1)[0]
            loss_query = self.alpha*self.balanced_continuous_focal_loss(center_score_list, GC_target)
            losses_query = losses_query + loss_query
        
        return dict(loss_query=losses_query)

    def balanced_continuous_focal_loss(self, inputs, targets):
        prob = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        scale = (prob - targets).abs() ** self.gamma
        loss = ce_loss * scale

        tau = torc.tensor(self.tau).cuda()
        weight = 1 / (1 + torch.exp(tau * (targets - 0.5)))
        loss = weight * loss
        return torch.mean(loss)

    def forward_test(self,
                      x, 
                      gt_userinputs,
                      x_userinput, 
                      img_metas=None,
                      **kwargs):
        bh = len(img_metas)
        imgs_scores = self(x)
        imgs_scores = imgs_scores.permute(0, 2, 3, 1).reshape(
                len(img_metas), -1, self.cls_out_channels).sigmoid() 
        max_score_imgs = torch.max(imgs_scores, dim=-1)
        sim_mask = (max_score_imgs[0] > self.threshold)

        f_size = [(int(img_metas[0]['ori_shape'][0]/4), int(img_metas[0]['ori_shape'][1]/4))]
        points = self.prior_generator.grid_priors(
                    f_size,
                    dtype=imgs_scores.dtype,
                    device=imgs_scores.device)[0]


        tensor1_new = torch.full((bh, 30), -1.0, device=gt_userinputs[0].device)  
        tensor2_new = torch.full((bh, 30, 2), -1.0, device=gt_userinputs[0].device) 
        tensor3_new = torch.full((bh, 30), -1.0, device=gt_userinputs[0].device) 
        tensor1_new[:, :20] = gt_userinputs[0]
        tensor2_new[:, :20, :] = gt_userinputs[1]
        tensor3_new[:, :20] = gt_userinputs[2]
        new_gt_userinputs = [tensor1_new, tensor2_new, tensor3_new]


        for img_id in range(len(img_metas)):
            filter_scores_mask = sim_mask[img_id]

            if filter_scores_mask.any():
                class_ids = gt_userinputs[2][img_id][gt_userinputs[0][img_id] !=-1].float()
                click_num = len(class_ids)

                top_k = min(10, sum(filter_scores_mask))
                center_topk_id = torch.topk(max_score_imgs[0][img_id][filter_scores_mask], top_k)[1]

                new_clicks = points[filter_scores_mask][center_topk_id]
                new_clicks_cls =max_score_imgs[1][img_id][filter_scores_mask][center_topk_id]
                unique_n_cls = torch.unique(new_clicks_cls)
                
                new_gt_userinputs[0][img_id, click_num:click_num+len(new_clicks)] = -2
                new_gt_userinputs[1][img_id, click_num:click_num+len(new_clicks)] = new_clicks
                new_gt_userinputs[2][img_id, click_num:click_num+len(new_clicks)] = new_clicks_cls
                for cls in unique_n_cls:
                    cls_gs_map = self.generate_weighted_mask(new_clicks[new_clicks_cls==cls], img_metas[0]['ori_shape'][:2])
                    x_userinput[img_id, cls] = torch.max(cls_gs_map, x_userinput[img_id, cls])
        return new_gt_userinputs, x_userinput
    
    def generate_weighted_mask(self, centre, shape, sigma=1.0):
        """Generate heatmap with multiple 2D gaussians using PyTorch tensors."""
        xs = torch.arange(0.5, shape[1] + 0.5, step=1.0, dtype=torch.float32, device=centre.device)
        ys = torch.arange(0.5, shape[0] + 0.5, step=1.0, dtype=torch.float32, device=centre.device).unsqueeze(-1)
        alpha = -0.5 / (sigma**2)
        heatmap = torch.zeros((centre.shape[0], shape[0], shape[1]), dtype=torch.float32, device=centre.device)
        for i in range(centre.shape[0]):
            x_center, y_center = centre[i]
            single_gaussian = torch.exp(alpha * ((xs - x_center)**2 + (ys - y_center)**2))
            heatmap[i] = single_gaussian
        heatmap = torch.max(heatmap, dim=0).values
        return heatmap

