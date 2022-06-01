import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.cm
from torchinfo import summary
import copy
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)
from mmdet.core.visualization import imshow_det_bboxes

device = 'cpu'
config = r"/home/ZhangYumeng/Workspace/mycode/mmdetection/configs/swin/cascade_rcnn_r50_fpn_fp16_ms-crop-3x_coco_dota.py"
ckpt = r"/home/ZhangYumeng/Workspace/mycode/mmdetection/tools/epoch_36.pth"
img = r"/home/ZhangYumeng/Workspace/mycode/mmdetection/tools/P2802__1__3296___4120.png"
out_dir = '/home/ZhangYumeng/Workspace/mycode/mmdetection/tools/out_dir/'
score_thr = 0.3
color_map = [[0, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
             [0, 0, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]
model = init_detector(config, ckpt, device=device)

module_name = []
p_in = []
p_out = []


def hook_fn(module, inputs, outputs):
    print(module_name)
    module_name.append(module.__class__)
    p_in.append(inputs)
    p_out.append(outputs)


def show_result_customized(model, img, result, score_thr, save_path):
    frame = cv2.imread(img)
    print(type(frame))
    result_sifted = []
    for i in range(len(result)):
        object_class_sifted = []
        for j in range(len(result[i])):
            if result[i][j][4] >= score_thr:
                object_class_sifted.append(result[i][j])

                frame = cv2.rectangle(frame,  # 图片
                                      (int(result[i][j][0]), int(result[i][j][1])),  # (xmin, ymin)左上角坐标
                                      (int(result[i][j][2]), int(result[i][j][3])),  # (xmax, ymax)右下角坐标
                                      (color_map[i][0] * 255, color_map[i][1] * 255, color_map[i][2] * 255), 3)

        result_sifted.append(object_class_sifted)
        print(type(frame))
        cv2.imwrite(save_path + 'show_result.png', np.array(frame, dtype='uint8'))


# print(model)

# model.register_forward_hook(hook_fn)
# for item in dir(model.backbone):
#     print(item)

model.backbone.register_forward_hook(hook_fn)
result = inference_detector(model, img)
# model.show_result(img, result, out_file='result.jpg')
show_result_customized(model, img, result, score_thr=score_thr, save_path=out_dir)
print("Draw BBox Succeed")


# def show_feature_map(img_src, conv_features, layer_num):
#    '''可视化卷积层特征图输出
#    img_src:源图像文件路径
#    conv_feature:得到的卷积输出,[b, c, h, w]
#    '''
#    img = Image.open(img_src).convert('RGB')
#    height, width = img.size
#    conv_features = conv_features.cpu()
#
#    heat = conv_features.squeeze(0)
#    heatmap = torch.mean(heat, dim=0)
#
#    heatmap = heatmap.numpy()
#    heatmap = np.maximum(heatmap, 0)
#    heatmap /= np.max(heatmap)
#    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
#    heatmap = np.uint8(255 * heatmap)
#    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
#
#    cv2.imwrite(out_dir + 'heatmap' + str(layer_num) + '.png', heatmap[:, :, ::-1])
#    plt.imshow(heatmap)
#    plt.show()
#    # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
#    superimg = heatmap * 0.4 + np.array(img)[:, :, ::-1]
#    cv2.imwrite(out_dir + 'superimg' + str(layer_num) + '.png', superimg)
#
#    img_ = np.array(Image.open(out_dir + 'superimg' + str(layer_num) + '.png').convert('RGB'))
#    plt.imshow(img_)
#    plt.show()

def show_feature_map(img_src, conv_features, layer_num):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    img = Image.open(img_src).convert('RGB')
    height, width = img.size
    conv_features = conv_features.cpu()

    heat = conv_features.squeeze(0)
    heatmap = torch.mean(heat, dim=0)

    heatmap = heatmap.numpy()
    div = np.max(heatmap) - np.min(heatmap)
    heatmap = heatmap + np.min(heatmap)
    heatmap = heatmap / div
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)

    cv2.imwrite(out_dir + 'heatmap' + str(layer_num) + '.png', heatmap[:, :, ::-1])
    plt.imshow(heatmap)
    plt.show()
    # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
    superimg = heatmap * 0.4 + np.array(img)[:, :, ::-1]
    cv2.imwrite(out_dir + 'superimg' + str(layer_num) + '.png', superimg)

    img_ = np.array(Image.open(out_dir + 'superimg' + str(layer_num) + '.png').convert('RGB'))
    plt.imshow(img_)
    plt.show()


for k in range(len(p_out[0])):
    show_feature_map(img, p_out[0][k], k)
