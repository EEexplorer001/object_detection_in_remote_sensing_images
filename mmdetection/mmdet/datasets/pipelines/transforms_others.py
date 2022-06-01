import cv2
import numpy as np

from mmdet.core import rotated_box_to_poly_np
from ..builder import PIPELINES

from .visualize import *
import os
import sys


@PIPELINES.register_module()
class HSVRandomAug:
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.
    Args:
        hue_delta (int): delta of hue. Default: 5.
        saturation_delta (int): delta of saturation. Default: 30.
        value_delta (int): delat of value. Default: 30.
    """

    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, results):
        img = results['img']
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img.astype(np.float32))

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


@PIPELINES.register_module()
class GtFilter(object):
    def __init__(self, max_per_img = 0, min_wh = 10, min_s = 80):
        self.max_per_img = max_per_img
        self.min_wh = min_wh
        self.min_s = min_s

    def whs_keep(self, bboxes):
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        s = w * h
        keep_inds = np.logical_and(
            np.logical_or(w >= self.min_wh, h >= self.min_wh),
            s >= self.min_s)
        return keep_inds

    def random_keep(self, bboxes):
        num = len(bboxes)
        if self.max_per_img != 0 and num > self.max_per_img:
            keep_inds = np.random.permutation(num)[:self.max_per_img]
            return keep_inds
        return np.asarray([True] * num)

    def bboxes_cat(self, bboxes1, bboxes2):
        if len(bboxes1) == 0:
            return bboxes2
        if len(bboxes2) == 0:
            return bboxes1
        return np.concatenate((bboxes1, bboxes2))

    def __call__(self, results):
        gt_bboxes_ignore = results.get('gt_bboxes_ignore', [])
        gt_bboxes        = results.get('gt_bboxes', [])
        gt_labels_ignore = results.get('gt_labels_ignore', [])
        gt_labels        = results.get('gt_labels', [])

        keep_inds = self.whs_keep(gt_bboxes)
        ignore_inds = np.logical_not(keep_inds)
        gt_bboxes_ignore = self.bboxes_cat(gt_bboxes_ignore, gt_bboxes[ignore_inds])
        gt_bboxes = gt_bboxes[keep_inds]
        gt_labels_ignore = self.bboxes_cat(gt_labels_ignore, gt_labels[ignore_inds])
        gt_labels = gt_labels[keep_inds]

        keep_inds = self.random_keep(gt_bboxes)
        ignore_inds = np.logical_not(keep_inds)
        gt_bboxes_ignore = self.bboxes_cat(gt_bboxes_ignore, gt_bboxes[ignore_inds])
        gt_bboxes = gt_bboxes[keep_inds]
        gt_labels_ignore = self.bboxes_cat(gt_labels_ignore, gt_labels[ignore_inds])
        gt_labels = gt_labels[keep_inds]

        results['gt_bboxes_ignore'] = gt_bboxes_ignore
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels_ignore'] = gt_labels_ignore
        results['gt_labels'] = gt_labels
        return results


@PIPELINES.register_module()
class TempVisualize(object):
    def __init__(self, note = "", img_rewrite = False, sys_exit = False):

        self.classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                        'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')
        self.work_dirs = "/home/ubuntu/Workspace/YuHongtian/mmdetection-master/work_dirs/"
        self.note = note
        self.img_rewrite = img_rewrite
        self.sys_exit = sys_exit

    def __call__(self, results):
        # WARNING: visualize_boxes() would not change boxes but would change image directly
        img = (results['img']).copy()
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])
        ori_filename = (results['ori_filename'])[:-4]

        if gt_bboxes.shape[-1] == 5:
            boxes = rotated_box_to_poly_np(gt_bboxes)
        elif gt_bboxes.shape[-1] == 4 or gt_bboxes.shape[-1] == 8:
            boxes = gt_bboxes
        else:
            raise ValueError(f"Invalid length of gt_bboxes")

        visualize_boxes(image = img, boxes = boxes, labels = labels, probs = np.ones(boxes.shape[0]), class_labels = self.classes)
        path = os.path.join(self.work_dirs, str(ori_filename) + "-" + str(self.note) + ".png")
        cv2.imwrite(path, img)

        if self.img_rewrite:
            results['img'] = img
        if self.sys_exit:
            sys.exit(0)

        return results