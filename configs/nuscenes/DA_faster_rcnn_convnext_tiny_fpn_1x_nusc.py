_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# import customer-defined modules and modules from mmcls
custom_imports = dict(imports=['my_modules.my_pipelines',
                               'my_modules.my_datasets',
                               'my_modules.my_detectors',
                               'mmcls.models.backbones',
                               'mmcls.models.necks',
                               'mmcls.models.heads'],
                      allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

# dataset settings
dataset_type = 'MyDataset'  # based on Coco dataset, add domain labels
data_root = r'E:/datasets/nuScenes/Mini dataset/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='DALoadAnnotations', domain_labels='rain_label', with_bbox=True, with_label=True),
    # dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),  # original image size is (1600, 900)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'rain_label']),  # add domain labels
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        domain_labels=['rain_label'],  # domain_labels for domain classification
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,  # classes used in training process
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(interval=2, metric=['bbox', 'proposal'])

# model settings
num_classes = len(class_names)  # num_classes of the dataset
model = dict(
    type='DATwoStageDetector',
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='tiny',
        in_channels=3,
        stem_patch_size=4,
        norm_cfg=dict(type='LN2d', eps=1e-6),
        act_cfg=dict(type='GELU'),
        linear_pw_conv=True,
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        frozen_stages=0,
        gap_before_final_norm=False,
        # init_cfg=[
        #     dict(
        #         type='TruncNormal',
        #         layer=['Conv2d', 'Linear'],
        #         std=.02,
        #         bias=0.),
        #     dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.)]),
        init_cfg=dict(
            type='Pretrained',
            # checkpoint=r'D:\PycharmProjects\pythonProject\mmdetection-master\checkpoints\backbones\convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
            checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # the same as backbone.out_channels(backbone.out_indices)
        out_channels=256,
        num_outs=5),  # use max pool to get more levels on top of outputs, refers to mmdet/models/necks/fpn.py
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,  # rpn_head.in_channels is the same as neck.out_channels
        feat_channels=256,  # out_channels of hidden layers in shared convs_layer before rpn_cls and rpn_reg in RPN Head
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),  # strides for generating anchors, also used as base_sizes
        # base_sizes: 1 pixel of single fpn output level refers to pixel number in original image
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,  # roi_head.bbox_roi_extractor.out_channels is the same as neck.out_channels
            featmap_strides=[4, 8, 16, 32]),  # chosed from rpn_head.anchor_generator.strides
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,  # roi_head.bbox_head.in_channels is the same as roi_head.bbox_roi_extractor.out_channels
            roi_feat_size=7,  # rois get through avg_pool to size (roi_feat_size, roi_feat_size)
            fc_out_channels=1024,  # flatten and get through 2FC
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=None,  # bbox_head and mask_head shares the same roi_extractor
        mask_head=None),
    # domain classifier
    domain_labels=['rain_label'],  # names of domain labels
    alpha=1.0,  # hyperparameter in GRL
    domain_neck=dict(type='mmcls.GlobalAveragePooling'),
    domain_head=dict(
        type='mmcls.LinearClsHead',
        num_classes=1,
        in_channels=256,
        # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0)),  # default
        loss=dict(type='FocalLoss', gamma=5.0, alpha=0.5, loss_weight=1.0)),  # focal loss means weak global alignment
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
