# By using the command
# >>> python ./tools/test.py ./configs/***.py work_dirs/***/latest.pth --out ***.pkl --eval bbox
# we will get a file named ***.pkl
#
# Now we should read this file and change the result into the format introduced in
# https://captain-whu.github.io/DOTA/tasks.html
# so that we can use DOTA_devkit/ResultMerge.py to merge the result
#

import os
import cv2
import pickle
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

from ResultMerge import mergebyrec, mergebypoly
from dota_evaluation_task2 import voc_eval

from visualize import *


class HBB_pkl_reader(object):
    """
    """
    def __init__(self, work_root, pkl_name, dataset_root, nms_thresh = 0.3, mode = 0):
        # 工作路径
        self.work_root = work_root
        # 工作路径下的子路径
        self.saveDir = self.work_root + "Task2/"
        self.mergeDir = self.work_root + "Task2_merge/"
        self.visualDir = self.work_root + "visual/"
        self.resultPickle = self.work_root + pkl_name

        # remakedirs
        if os.path.exists(self.saveDir):
            shutil.rmtree(self.saveDir)
        if os.path.exists(self.mergeDir):
            shutil.rmtree(self.mergeDir)
        if os.path.exists(self.visualDir):
            shutil.rmtree(self.visualDir)

        os.makedirs(self.saveDir)
        os.makedirs(self.mergeDir)
        os.makedirs(self.visualDir)

        # merge前的所有类别的txt文件
        self.split_detpath = os.path.join(self.saveDir, 'Task2_{:s}.txt')
        # merge后的所有类别的txt文件
        self.detpath = os.path.join(self.mergeDir, 'Task2_{:s}.txt')

        # 数据集根路径
        self.dataset_root = dataset_root
        # 验证集的原始标注txt文件
        self.annopath = self.dataset_root + r'DOTA/val/labelTxt/{:s}.txt'
        # 验证集的原始图片名文件
        self.imagesetfile = self.dataset_root + r'DOTA/val/valset.txt'
        # 分割前的图片路径
        self.imagepath = self.dataset_root + r'DOTA/val/images/'
        # 分割后的图片路径
        self.split_imagepath = self.dataset_root + r'DOTA-ImgSplit/val/images/'
        # 分割后的json文件路径
        self.split_jsonpath = self.dataset_root + r'DOTA-ImgSplit-COCO/annotations/instances_val2017.json'
        # 分割后的标注txt文件
        self.split_annopath = self.dataset_root + r'DOTA-ImgSplit/val/labelTxt/{:s}.txt'
        # 分割后的图片名文件
        self.split_imagesetfile = self.dataset_root + r'DOTA-ImgSplit/val/valset.txt'

        self.classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                           'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

        self.val_coco = COCO(self.split_jsonpath) # 读取json文件
        self.result_all = pickle.load(open(self.resultPickle, 'rb')) # 读取pkl文件

        self.cats = self.val_coco.loadCats(self.val_coco.getCatIds())
        self.cat_labels = [cat["name"] for cat in self.cats]
        self.cat_inds = [cat["id"] for cat in self.cats]
        self.ind_to_label_map = {ind: label for ind, label in zip(self.cat_inds, self.cat_labels)}

        # 至此，ind_to_label_map的结果是
        # {1: "plane", 2: "baseball-diamond", ..., 15: "helicopter"}

        self.imgs = self.val_coco.loadImgs(self.val_coco.getImgIds())
        self.img_imgnames = [img["file_name"] for img in self.imgs]
        self.img_inds = [img["id"] for img in self.imgs]
        self.ind_to_imgname_map = {ind: imgname for ind, imgname in zip(self.img_inds, self.img_imgnames)}

        # 至此，ind_to_imgname_map的结果是
        # {1: "P1156__1__1648___824.png", 2: "P1513__1__0___2472.png", ..., 5297: "P1699__1__1648___2472.png"}

        self.nms_thresh = nms_thresh
        self.mode = mode

    def write_split_imagesetfile(self):
        # 把ind_to_imagefile_map的value写到split_imagesetfile里去
        with open(self.split_imagesetfile, 'w') as f:
            for imgname in self.img_imgnames:
                f.write(imgname[:-4] + '\n')

    def write_saveDir(self):
        # 把未merge的结果写到saveDir下
        for cat_id in self.ind_to_label_map.keys():
            with open(os.path.join(self.saveDir, "Task2_" + self.ind_to_label_map[cat_id] + ".txt"), 'w') as f:
                for image_id, result_per_image in enumerate(self.result_all):
                    cat_bboxes = result_per_image[cat_id - 1]
                    for det in cat_bboxes:
                        bbox = det[-5:-1]
                        score = str(round(det[-1], 5))
                        imgname = self.ind_to_imgname_map[image_id + 1][:-4] # 去掉拓展名

                        xmin_str = str(int(bbox[0]))
                        ymin_str = str(int(bbox[1]))
                        xmax_str = str(int(bbox[2]))
                        ymax_str = str(int(bbox[3]))

                        f.write(imgname + ' ' + score + ' ' +
                                xmin_str + ' ' + ymin_str + ' ' +
                                xmax_str + ' ' + ymax_str + ' ' + '\n')

    def visualize_saveDir(self):
        # saveDir的可视化
        print("\nNOW VISUALIZING SAVE_DIR...")
        ind = 1
        for result_per_image in tqdm(self.result_all):
            img_path = os.path.join(self.split_imagepath, self.ind_to_imgname_map[ind])
            img = cv2.imread(img_path)
            bboxes_prd = np.zeros((0, 6))
            for cat_id in self.ind_to_label_map.keys():
                cat_bboxes = result_per_image[cat_id - 1]
                for det in cat_bboxes:
                    bboxes_new = np.append(det[-5:], cat_id - 1)
                    bboxes_new = np.expand_dims(bboxes_new, axis = 0)
                    bboxes_prd = np.concatenate((bboxes_prd, bboxes_new), axis = 0)

            boxes = bboxes_prd[..., :4]
            class_inds = bboxes_prd[..., 5].astype(np.int32)
            scores = bboxes_prd[..., 4]

            visualize_boxes(image = img, boxes = boxes, labels = class_inds, probs = scores, class_labels = self.classnames)
            path = os.path.join(self.visualDir, self.ind_to_imgname_map[ind])
            cv2.imwrite(path, img)
            ind += 1

    def write_mergeDir(self):
        # 利用dota工具包进行merge，将每一类在split上的检测结果合并到大图上
        # merge工具自带偏移量计算
        # task1: mergebypoly
        # task2: mergebyrec
        print("\nNOW MERGING...")
        mergebyrec(self.saveDir, self.mergeDir, self.nms_thresh)

    def visualize_mergeDir(self):
        # mergeDir的可视化
        print("\nNOW VISUALIZING MERGE_DIR...")
        imgname_xyxysc_dict = {}

        for cat_id in self.ind_to_label_map.keys():
            with open(os.path.join(self.mergeDir, "Task2_" + self.ind_to_label_map[cat_id] + ".txt"), 'r') as f:
                print("Working with Task2_" + self.ind_to_label_map[cat_id] + ".txt")
                lines = f.readlines()
                splitlines = [x.strip().split(' ')  for x in lines]
                for splitline in tqdm(splitlines):
                    if (len(splitline) != 6):
                        continue
                    imgname = splitline[0]
                    bboxes_new = np.asarray([float(splitline[2]),
                                             float(splitline[3]),
                                             float(splitline[4]),
                                             float(splitline[5]),
                                             float(splitline[1]),
                                             float(cat_id - 1)])
                    bboxes_new = np.expand_dims(bboxes_new, axis = 0)
                    if (imgname not in imgname_xyxysc_dict):
                        imgname_xyxysc_dict[imgname] = bboxes_new
                    else:
                        bboxes_prd = imgname_xyxysc_dict[imgname]
                        imgname_xyxysc_dict[imgname] = np.concatenate((bboxes_prd, bboxes_new), axis = 0)

        for imgname in tqdm(imgname_xyxysc_dict.keys()):
            img_path = os.path.join(self.imagepath, imgname + ".png")
            img = cv2.imread(img_path)
            bboxes_prd = imgname_xyxysc_dict[imgname]
            boxes = bboxes_prd[..., :4]
            class_inds = bboxes_prd[..., 5].astype(np.int32)
            scores = bboxes_prd[..., 4]

            visualize_boxes(image = img, boxes = boxes, labels = class_inds, probs = scores, class_labels = self.classnames)
            path = os.path.join(self.visualDir, imgname + ".png")
            cv2.imwrite(path, img)

    def evaluate(self):
        # 进行voc评测
        # 照抄dota_evaluation_task2的main函数即可
        print("\nNOW EVALUATING AP50...")
        classaps = []
        map = 0
        for classname in self.classnames:
            print('classname:', classname)
            rec, prec, ap = voc_eval(self.detpath,
                    self.annopath,
                    self.imagesetfile,
                    classname,
                    ovthresh=0.5,
                    use_07_metric=False)
            map = map + ap
            #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            print('ap: ', ap)
            classaps.append(ap)

            ## uncomment to plot p-r curve for each category
            # plt.figure(figsize=(8,4))
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.plot(rec, prec)
            # plt.show()
        map = map/len(self.classnames)
        print('map:', map)
        classaps = 100*np.array(classaps)
        print('classaps: ', classaps)

    def main(self):
        # self.write_split_imagesetfile()

        self.write_saveDir()
        if self.mode == 1 or self.mode == 3:
            self.visualize_saveDir()

        self.write_mergeDir()
        if self.mode == 2 or self.mode == 3:
            self.visualize_mergeDir()

        self.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_root", type = str, default = "/home/ZhangYumeng/Workspace/DOTA_devkit-master/DOTA_devkit-master/", help = "work root")
    parser.add_argument("--pkl_name", type = str, default = "xxx.pkl", help = "e.g. xxx.pkl")
    parser.add_argument("--dataset_root", type = str, default = "/home/ZhangYumeng/Dataset/", help = "dataset root")
    parser.add_argument("--nms_thresh", type = float, default = 0.5, help = "nms_thresh when merge")
    parser.add_argument("--mode", type = int, default = 0, help = "visualize mergeDir|saveDir: 0-00, 1-01, 2-10, 3-11")
    opt = parser.parse_args()

    HBB_pkl_reader(work_root = opt.work_root, pkl_name = opt.pkl_name, dataset_root = opt.dataset_root, nms_thresh = opt.nms_thresh, mode = opt.mode).main()