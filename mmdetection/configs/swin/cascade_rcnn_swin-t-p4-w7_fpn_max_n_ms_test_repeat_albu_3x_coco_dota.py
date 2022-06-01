_base_ = './cascade_rcnn_r50_fpn_1x_coco_dota.py'
dataset_type = 'Dota2CocoDataset'
data_root = '/home/ubuntu/Dataset/DOTA-ImgSplit-COCO/'

# from './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py' ##### #####


# model = dict(
#     backbone=dict(
#         depth=101,
#         init_cfg=dict(type='Pretrained',
#                       checkpoint='torchvision://resnet101')))


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]),
    test_cfg=dict(
        rcnn=dict(
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=1000)))


img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[81.93, 82.84, 78.56], std=[47.23, 45.73, 44.41], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# from './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py' ##### #####

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=(-90,90),
        interpolation=1,
        p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # ver 1.00
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # 
    # ver 1.01
    # dict(type='BoxToBox', mode='rotated_box_to_bbox_np'),
    dict(type='RandomFlip', flip_ratio=[0.3, 0.3], direction = ['horizontal', 'vertical']),
    # 
    
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline)
            )