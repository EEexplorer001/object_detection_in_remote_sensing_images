_base_ = './cascade_rcnn_r50_fpn_1x_coco_dota.py'


# from './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py' ##### #####


# model = dict(
#     backbone=dict(
#         depth=101,
#         init_cfg=dict(type='Pretrained',
#                       checkpoint='torchvision://resnet101')))


img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[81.93, 82.84, 78.56], std=[47.23, 45.73, 44.41], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)

# from './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py' ##### #####


# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))