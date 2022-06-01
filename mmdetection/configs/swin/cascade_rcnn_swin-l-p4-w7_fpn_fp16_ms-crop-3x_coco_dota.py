_base_ = './cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py'


# from https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py ##### #####


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k_finetune.pth'
# noqa
model = dict(
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]))


# recommend: batch_size=16, init_lr=1e-4
# now with 8x12GB GPU: batch_size=8, init_lr=5e-5
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)
optimizer = dict(lr=5e-5)