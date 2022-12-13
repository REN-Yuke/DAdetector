# import customer-defined modules and modules from mmcls
custom_imports = dict(imports=['my_modules.detectors',
                               'mmcls.models.necks',
                               'mmcls.models.heads'],
                      allow_failed_imports=False)

# model settings
num_classes = 10  # num_classes of the dataset
model = dict(
    type='DATwoStageDetector',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,  # number of res_layers
        out_indices=(0, 1, 2, 3),  # except stem, output all res_layers(layer1, layer2, layer3, layer4)
        frozen_stages=1,  # frozen stem and layer1
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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
        num_classes=2,  # 2 classes: rain_label=0 for images without rain, rain_label=1 for images with rain
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
