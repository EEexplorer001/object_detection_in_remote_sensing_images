_base_ = './cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py'


# from './mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py' ##### #####


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
# noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))


# recommend: batch_size=16, init_lr=1e-4
# now with 8x12GB GPU: batch_size=8, init_lr=5e-5
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)
optimizer = dict(lr=5e-5)