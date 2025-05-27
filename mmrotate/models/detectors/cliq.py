import warnings

import logging
from mmcv.runner import get_dist_info
import numpy as np

from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_neck

from .rotated_fcos import RotatedFCOS
# from ..necks.feat_outaug import featoutaug
import torch
import torch.nn as nn
import torch.nn.functional as F

@ROTATED_DETECTORS.register_module()
class CLIQ(RotatedFCOS):

    def __init__(self,
                 backbone,
                 neck,
                 isq=None,
                 icq=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 userinput_backbone=None,
                 userinput_neck=None):
        super(CLIQ, self).__init__(backbone, neck, bbox_head,
                                             train_cfg, test_cfg, pretrained, init_cfg)

        self.isq = build_neck(isq)
        self.icq = build_neck(icq)
        self.bbox_head_cfg = bbox_head
        self.num_classes = self.bbox_head_cfg['num_classes']

        if userinput_backbone is not None:
            self.userinput_backbone = build_backbone(userinput_backbone)
            if userinput_neck is not None:
                self.with_userinput_neck = True
                self.userinput_neck = build_neck(userinput_neck)

        self.conv = nn.Conv2d(in_channels=self.bbox_head_cfg['in_channels']*2  +self.bbox_head_cfg['num_classes'], 
                                out_channels=self.bbox_head_cfg['in_channels'], kernel_size=1)

    def set_epoch(self, epoch): 
        self.bbox_head.epoch = epoch 
        self.isq.epoch = epoch 
        self.epoch = epoch 


    def extract_userinput_feat(self, userinput):
        x = self.userinput_backbone(userinput)
        if self.with_userinput_neck:
            x = self.userinput_neck(x)
        return x

    def process_userinput(self, img, **kwargs):
        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        img, user_input = img[:, :3, :, :], img[:, 3:, :, :]
        f = self.extract_feat(img)
        f_userinput = self.extract_userinput_feat(user_input)

        return f, f_userinput
        
    def forward_train(self,
                img,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=None,
                **kwargs):
        kwargs.update({'img_metas': img_metas})
        losses = dict()

        gt_userinputs = kwargs['gt_userinputs']
        x, x_userinput = self.process_userinput(img, **kwargs)

        query_loss = self.icq.forward_train(x, gt_bboxes, gt_labels, img_metas)     
        losses.update(query_loss)

        x, x_userinput = self.isq(x, gt_userinputs, img_metas, x_userinput)
        x = [self.conv(torch.cat((feat, user_input_feat), dim=1)) 
             for feat, user_input_feat in zip(x, x_userinput)]
        bbox_losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,gt_labels, 
                                        gt_bboxes_ignore)
        losses.update(bbox_losses)


        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        kwargs.update({'img_metas': img_metas})
        if img.dtype != torch.float32:
            img = img.to(torch.float32)
        gt_userinputs = kwargs['gt_userinputs']
        img, user_input = img[:, :3, :, :], img[:, 3:, :, :]
        img_feat = self.extract_feat(img)

        gt_userinputs, user_input = self.icq.forward_test(img_feat, gt_userinputs, user_input, img_metas)        
        x_userinput = self.extract_userinput_feat(user_input)

        x, x_userinput = self.isq(img_feat, gt_userinputs, img_metas, x_userinput)
        x = [self.conv(torch.cat((feat, user_input_feat), 1)) 
             for feat, user_input_feat in zip(x, x_userinput)]

        rbbox_outs = self.bbox_head(x)

        bbox_list = self.bbox_head.get_bboxes(
            *rbbox_outs, img_metas, rescale=rescale)
        
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
