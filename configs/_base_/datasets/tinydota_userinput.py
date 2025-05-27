# dataset settings
img_size=1024

dataset_type = 'TinyDOTADatasetUserInput'
data_root = 'path-to-data'
config_max_user_input = 20
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(img_size, img_size)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        max_user_input = config_max_user_input,
        max_gt_num_perimg=500,
        rand_sample = None,
        rand_ratio = 0.9,
        version='le90',
        ann_file=data_root + 'train/labelTxt',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        ori_ann_file='path-to-raw-annotation-files'),
    val=dict(
        type=dataset_type,
        max_user_input = config_max_user_input,
        max_gt_num_perimg=2000,
        rand_sample = None,
        version='le90',
        ann_file=data_root + 'test/labelTxt',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        ori_ann_file='path-to-raw-annotation-files'
        ),
    test=dict(
        type=dataset_type,
        max_user_input = config_max_user_input,
        max_gt_num_perimg=2000,
        rand_sample = None,
        rand_ratio = 0.9,
        version='le90',
        ann_file=data_root + 'test/labelTxt',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        ori_ann_file='path-to-raw-annotation-files'
        )
            )
