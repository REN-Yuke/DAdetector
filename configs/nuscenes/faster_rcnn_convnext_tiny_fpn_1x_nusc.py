_base_ = [
    '../_base_/datasets/nucenes_2d_detection.py',
    '../_base_/models/faster_rcnn_convnext_tiny_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# For nuScenes, num_classes=10
model = dict(roi_head=dict(bbox_head=dict(num_classes=10)))

