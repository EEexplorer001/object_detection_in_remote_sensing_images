machine="t1"
note="cosine"
epoch=3
while [ $epoch -le 12 ]
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/swin/cascade_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-1x_cos-e_coco_dota.py "./work_dirs/cascade_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-1x_cos-e_coco_dota/epoch_${epoch}.pth" 8 --out "/home/ZhangYumeng/Workspace/mycode/DOTA_devkit-master/${machine}-${note}-${epoch}.pkl" --eval bbox
    let epoch++
done
