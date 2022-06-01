_base_ = './cascade_rcnn_r50_fpn_1x_coco_dota.py'


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

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)

# from './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py' ##### #####



# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))