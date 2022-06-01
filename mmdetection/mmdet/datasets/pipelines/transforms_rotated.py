# 
# From https://github.com/csuhan/s2anet/blob/master/mmdet/datasets/pipelines/transforms_rotated.py
# 
# 1. from ..builder import PIPELINES
# 2. @PIPELINES.register_module()
# 3. rewrite class RotatedRandomFlip
# 4. rate -> rotate_ratio, angles -> angles & angles_type
# 5. remove auto_bound
# 6. new class BoxToBox
# 

import random

import cv2
import numpy as np

from mmdet.core import poly_to_rotated_box_np, rotated_box_to_poly_np, norm_angle, bbox_to_rotated_box_np, rotated_box_to_bbox_np
from .transforms import RandomFlip, Resize
from ..builder import PIPELINES


@PIPELINES.register_module()
class RotatedRandomFlip(RandomFlip):

    #def bbox_flip(self, bboxes, img_shape):
    #    """Flip bboxes horizontally.

    #    Args:
    #        bboxes(ndarray): shape (..., 5*k)
    #        img_shape(tuple): (height, width)
    #    """
    #    assert bboxes.shape[-1] % 5 == 0
    #    w = img_shape[1]
    #    # x_ctr and angle
    #    bboxes[..., 0::5] = w - bboxes[..., 0::5] - 1
    #    bboxes[..., 4::5] = norm_angle(np.pi - bboxes[..., 4::5])

    #    return bboxes

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.
        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 5*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        if len(bboxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        assert bboxes.shape[-1] % 5 == 0
        if direction == 'horizontal':
            w = img_shape[1]
            bboxes[..., 0::5] = w - bboxes[..., 0::5]                   # x_ctr
            bboxes[..., 4::5] = norm_angle(np.pi - bboxes[..., 4::5])   # angle
        elif direction == 'vertical':
            h = img_shape[0]
            bboxes[..., 1::5] = h - bboxes[..., 1::5]                   # y_ctr
            bboxes[..., 4::5] = norm_angle(-bboxes[..., 4::5])          # angle
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            bboxes[..., 0::5] = w - bboxes[..., 0::5]                   # x_ctr
            bboxes[..., 1::5] = h - bboxes[..., 1::5]                   # y_ctr
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return bboxes


@PIPELINES.register_module()
class RotatedResize(Resize):

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            polys = rotated_box_to_poly_np(results[key])  # to 8 points
            polys = polys * results['scale_factor']
            if polys.shape[0] != 0:
                polys[:, 0::2] = np.clip(polys[:, 0::2], 0, img_shape[1] - 1)
                polys[:, 1::2] = np.clip(polys[:, 1::2], 0, img_shape[0] - 1)
            rboxes = poly_to_rotated_box_np(polys)  # to x,y,w,h,angle
            results[key] = rboxes


@PIPELINES.register_module()
class PesudoRotatedRandomFlip(RandomFlip):
    def __call__(self, results):
        return results


@PIPELINES.register_module()
class PesudoRotatedResize(Resize):
    def __call__(self, results):
        results['scale_factor'] = 1.0
        return results


@PIPELINES.register_module()
class RandomRotate(object):
    def __init__(self,
                 rotate_ratio=0.5,
                 angles=[0, 180],
                 angles_type='absolute_range'):
        #          auto_bound=False):
        self.rotate_ratio = rotate_ratio
        self.angles = angles
        self.angles_type = angles_type
        assert angles_type in ['absolute', 'absolute_range']
        # new image shape or not
        # self.auto_bound = auto_bound

    @property
    def rand_angle(self):
        if self.angles_type == 'absolute':
            return random.sample(self.angles, 1)[0]
        elif self.angles_type == 'absolute_range':
            return random.random() * (self.angles[1] - self.angles[0]) + self.angles[0]
        else:
            return 0

    @property
    def is_rotate(self):
        return random.random() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, center, angle, bound_h, bound_w, offset=0):
        center = (center[0] + offset, center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        #if self.auto_bound:
        #    # Find the coordinates of the center of rotation in the new image
        #    # The only point for which we know the future coordinates is the center of the image
        #    rot_im_center = cv2.transform(
        #        center[None, None, :] + offset, rm)[0, 0, :]
        #    new_center = np.array(
        #        [bound_w / 2, bound_h / 2]) + offset - rot_im_center
        #    # shift the rotation center to the new coordinates
        #    rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h)
        return keep_inds

    def __call__(self, results):
        # return the results directly if not rotate
        if not self.is_rotate:
            results['rotate'] = False
            return results

        h, w, c = results['img_shape']
        img = results['img']
        # angle for rotate
        angle = self.rand_angle
        results['rotate'] = True
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = abs(np.cos(angle)), abs(np.sin(angle))
        #if self.auto_bound:
        #    # find the new width and height bounds
        #    bound_w, bound_h = np.rint(
        #        [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
        #    ).astype(int)
        #else:
        bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w)
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)
        # rotate img
        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        # rotate bboxes
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])

        # no operation if gt_bboxes empty
        if len(gt_bboxes) == 0:
            return None
        # no rotation if angle is 0 or np.pi / 2
        # e.g. 'storage-tank', 'roundabout', etc.
        CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                   'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')
        eps = 1e-6
        gt_bboxes_angles = gt_bboxes[:, -1]
        gt_bboxes_nr_inds = np.logical_and(
            np.logical_or(labels == CLASSES.index('storage-tank'),
                          labels == CLASSES.index('roundabout')),
            np.logical_or(np.abs(gt_bboxes_angles) < eps,
                          np.abs(gt_bboxes_angles - np.pi / 2) < eps))
        # rotate in counter-clockwise direction so compensate in clockwise direction
        gt_bboxes[gt_bboxes_nr_inds, -1] = norm_angle(gt_bboxes[gt_bboxes_nr_inds, -1] + angle / 180 * np.pi)

        polys = rotated_box_to_poly_np(gt_bboxes).reshape(-1, 2)
        polys = self.apply_coords(polys).reshape(-1, 8)
        gt_bboxes = poly_to_rotated_box_np(polys)
        keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
        gt_bboxes = gt_bboxes[keep_inds, :]
        labels = labels[keep_inds]
        if len(gt_bboxes) == 0:
            return None
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = labels
        return results


@PIPELINES.register_module()
class BoxToBox(object):
    def __init__(self, mode = 'rotated_box_to_bbox_np'):
        valid_mode = ['rotated_box_to_bbox_np', 'bbox_to_rotated_box_np']
        assert mode in valid_mode

        if mode == 'rotated_box_to_bbox_np':
            self.shapein = 5
            self.shapeout = 4
            self.func = rotated_box_to_bbox_np
        elif mode == 'bbox_to_rotated_box_np':
            self.shapein = 4
            self.shapeout = 5
            self.func = bbox_to_rotated_box_np
        else:
            pass

    def transf(self, bboxes):
        if len(bboxes) == 0:
            return np.zeros((0, self.shapeout), dtype=np.float32)
        assert bboxes.shape[-1] % self.shapein == 0
        bboxes = self.func(bboxes)
        bboxes = np.float32(bboxes)
        return bboxes

    def __call__(self, results):
        gt_bboxes = results.get('gt_bboxes', [])
        gt_bboxes = self.transf(gt_bboxes)
        results['gt_bboxes'] = gt_bboxes

        gt_bboxes_ignore = results.get('gt_bboxes_ignore', [])
        gt_bboxes_ignore = self.transf(gt_bboxes_ignore)
        results['gt_bboxes_ignore'] = gt_bboxes_ignore

        return results