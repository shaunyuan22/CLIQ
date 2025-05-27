_base_ = [
    '../../_base_/datasets/tinydota_userinput.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

angle_version = 'le90'
find_unused_parameters=True
evaluation = dict(interval=12, metric='mAP')
runner = dict(type='EpochBasedRunner', max_epochs=12)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
data = dict(samples_per_gpu=1,
            train=dict(max_user_input = 20, rand_sample = None, max_gt_num_perimg=1000),
            val=dict(max_user_input = 20, rand_sample = None, max_gt_num_perimg=1000),
            test=dict(max_user_input = 20, rand_sample = None, max_gt_num_perimg=2000)#max_gt_num_perimg invalid in test
            )
lr_config = dict(warmup_iters=5000)
object_classes = 8

# model settings
model = dict(
    type='CLIQ',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    isq=dict(
        type='ISQ',
        feat_channels=256,
        cls_channels=object_classes,
        ),    
    icq=dict(
        type='ICQ',
        threshold=0.5,
        alpha=20.0,
        feat_channels=256,
        cls_channels=object_classes,
        n=0.5
        ), 
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=object_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000),
    userinput_backbone = dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        in_channels=object_classes,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    userinput_neck = dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

