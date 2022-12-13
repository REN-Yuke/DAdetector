# import customer-defined modules
custom_imports = dict(imports=['my_modules.my_pipelines',
                               'my_modules.my_datasets'],
                      allow_failed_imports=False)

# dataset settings
dataset_type = 'MyDataset'  # based on Coco dataset, add domain labels
# data_root = r'E:/datasets/nuScenes/Full dataset (v1.0)/trainval/'
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
        ann_file=data_root + 'mini_infos_train_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,  # classes used in training process
        pipeline=train_pipeline,
        test_mode=False,
        filter_empty_gt=False),  # set filter_empty_gt to False, training process uses negative positives
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mini_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mini_infos_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(interval=1, metric=['bbox', 'proposal'])
