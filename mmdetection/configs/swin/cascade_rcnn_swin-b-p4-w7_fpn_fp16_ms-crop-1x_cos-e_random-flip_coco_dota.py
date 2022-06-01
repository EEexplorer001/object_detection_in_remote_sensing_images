_base_ = './cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-1x_coco_dota.py'


# from https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py ##### #####


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'
# noqa
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512, 1024]))

img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[81.93, 82.84, 78.56], std=[47.23, 45.73, 44.41], to_rgb=True)
    
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # ver 1.00
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # 
    # ver 1.01
    # dict(type='BoxToBox', mode='rotated_box_to_bbox_np'),
    dict(type='RandomFlip', flip_ratio=[0.5, 0.5], direction = ['horizontal', 'vertical']),
    # 
   
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# lr_config = dict(_delete_=True, policy='CosineAnnealing', by_epoch=True, warmup='linear', warmup_iters=500, warmup_ratio=0.001, min_lr=0.)


# recommend: batch_size=16, init_lr=1e-4
# now with 8x12GB GPU: batch_size=8, init_lr=5e-5
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)
optimizer = dict(lr=5e-5)