import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, Linear
from mmcv.runner import BaseModule, auto_fp16
from ..builder import ROTATED_NECKS, build_loss

@ROTATED_NECKS.register_module()
class ISQ(BaseModule):

    def __init__(self,
                 feat_channels,
                 cls_channels=8,
                 alpha=0.1,
                 conv_bias='auto',
                 conv_cfg=None, 
                 norm_cfg=dict(type='BN', requires_grad=True),
):
        super(ISQ, self).__init__()

        self.feat_channels = feat_channels
        self.cls_channels = cls_channels
        self.alpha = alpha
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        self.K_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 1)
        self.V_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 1)
        self.Q_conv = nn.ModuleList()
        for i in range(self.cls_channels):
            self.Q_conv.append(nn.Conv2d(self.feat_channels, self.feat_channels, 1))

    def forward(self, feats, gt_userinputs, img_metas, f_userinputs):
        img_shape = img_metas[0]['img_shape'][:2]
        f_wh = [(feat.shape[2], feat.shape[3]) for feat in feats]
        new_feats = [torch.cat((feat, torch.zeros((feat.shape[0], self.cls_channels, feat.shape[2], feat.shape[3]), device=feat.device)), 
                            dim=1) 
                                for feat in feats] 

        new_f_userinputs = [f_userinput.clone() for f_userinput in f_userinputs] 

        cls_gs_maps = torch.zeros((len(img_metas), self.cls_channels, img_shape[0], img_shape[1]), device=feats[0].device)
        for img_idx in range(len(img_metas)):
            class_ids = gt_userinputs[2][img_idx][gt_userinputs[0][img_idx] !=-1].float()
            click_points = gt_userinputs[1][img_idx][0:len(class_ids)].float()
            if len(class_ids) == 0:
                continue
            click_classes = torch.unique(class_ids).to(torch.long)

            for click_class in click_classes:
                cls_clicks = click_points[class_ids == click_class]
                cls_clicks_heatmap = self.generate_weighted_mask(cls_clicks, img_shape)
                cls_gs_maps[img_idx, click_class] = cls_clicks_heatmap
                for level, (wh, fe) in enumerate(zip(f_wh, feats)):
                    down_sample_heatmap = F.interpolate(cls_clicks_heatmap.unsqueeze(0).unsqueeze(0), 
                                            size=wh).squeeze(0).squeeze(0) 
                    feat_weight = down_sample_heatmap * fe[img_idx]
                    cls_feat_weight = feat_weight.permute(1, 2, 0).reshape(-1, self.feat_channels).sum(0)
                    q_feat = self.Q_conv[click_class](cls_feat_weight.reshape(1, -1, 1, 1))
                    q_feat = torch.softmax(q_feat.reshape(-1), dim=0)
   
                    k_feat = self.K_conv(fe[img_idx].unsqueeze(0)).squeeze(0).permute(1,2,0).reshape(-1, self.feat_channels)

                    simile_mask = torch.matmul(k_feat, q_feat).reshape(wh[0],wh[1])

                    if self.training:
                        new_feats[level][img_idx, self.feat_channels+click_class] = simile_mask.sigmoid()
                    else:
                        new_feats[level][img_idx, self.feat_channels+click_class] = simile_mask

        # if self.training and self.epoch <= 2:
        #     return new_feats, new_f_userinputs

        for l, feat in enumerate(feats):
            v_feat = self.V_conv(feat)
            
            for img_idx in range(len(img_metas)):
                class_ids = gt_userinputs[2][img_idx][gt_userinputs[0][img_idx] !=-1].float()
                if len(class_ids) == 0:
                    continue
                down_sample_heatmap = F.interpolate(cls_gs_maps[img_idx].unsqueeze(0), 
                        size=f_wh[l]).squeeze(0)                
                heatmap_mask = (down_sample_heatmap > 0.).any(dim=0)
                if heatmap_mask.any():
                    if self.training:
                        simile_cls_mask = new_feats[l][img_idx, self.feat_channels:]
                    else:
                        simile_cls_mask = new_feats[l][img_idx, self.feat_channels:].sigmoid()

                    cls_click = torch.sum(simile_cls_mask, dim=(1, 2)) 

                    mean_mask = torch.mean(simile_cls_mask[cls_click>0], dim=0)
                    
                    new_feats[l][img_idx, :self.feat_channels] = mean_mask * v_feat[img_idx] + feat[img_idx]

        return new_feats, new_f_userinputs


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