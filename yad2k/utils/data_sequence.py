from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
import math
import cv2
import numpy as np
import os


class SequenceData(Sequence):

    def __init__(self, path, input_shape, batch_size, anchors, num_classes, max_boxes=20, shuffle=True):
        """
        初始化数据发生器
        :param path: 数据路径
        :param input_shape: 模型输入图片大小
        :param batch_size: 一个批次大小
        :param max_boxes: 一张图像中最多的box数量，不足的补充0， 超出的截取前max_boxes个，默认20
        :param shuffle: 数据乱序
        """
        # 1.打开文件
        self.datasets = []
        with open(path, "r")as f:
            self.datasets = f.readlines()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.anchors = anchors
        self.shuffle = shuffle
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.max_boxes = max_boxes

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_images = len(self.datasets)
        return math.ceil(num_images / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.datasets[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch)
        return X, y

    def get_epochs(self):
        return self.__len__()

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_true_boxes(self, boxes):
        """
        根据box来计算 detectors_mask 和 matching_true_boxes
        :param boxes:
            List of ground truth boxes in form of relative x, y, w, h, class.
            Relative coordinates are in the range [0, 1] indicating a percentage
            of the original image dimensions.
        :return:
        detectors_mask : array
            0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
            that should be compared with a matching ground truth box.
        matching_true_boxes: array
            Same shape as detectors_mask with the corresponding ground truth box
            adjusted for comparison with predicted parameters at training time.
        """
        height, width = self.input_shape
        num_anchors = len(self.anchors)
        # Downsampling factor of 5x 2-stride max_pools == 32.
        # TODO: Remove hardcoding of downscaling calculations.
        assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        conv_height = height // 32
        conv_width = width // 32
        num_box_params = boxes.shape[1]
        detectors_mask = np.zeros(
            (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
        matching_true_boxes = np.zeros(
            (conv_height, conv_width, num_anchors, num_box_params),
            dtype=np.float32)

        for box in boxes:
            # scale box to convolutional feature spatial dimensions
            box_class = box[4:5]
            box = box[0:4] * np.array(
                [conv_width, conv_height, conv_width, conv_height])
            i = np.floor(box[1]).astype('int')
            j = np.floor(box[0]).astype('int')
            best_iou = 0
            best_anchor = 0
            for k, anchor in enumerate(self.anchors):
                # Find IOU between box shifted to origin and anchor box.
                # 这里假设box和anchor中心点重叠，并且以目标中心点为坐标原点
                box_maxes = box[2:4] / 2.
                box_mins = -box_maxes
                anchor_maxes = (anchor / 2.)
                anchor_mins = -anchor_maxes
                # 计算实际box和anchor的iou，找到iou最大的anchor
                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]
                box_area = box[2] * box[3]
                anchor_area = anchor[0] * anchor[1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = k

            if best_iou > 0:
                # 置位iou最大的anchor
                detectors_mask[i, j, best_anchor] = 1
                adjusted_box = np.array(
                    [
                        box[0] - j, box[1] - i,
                        np.log(box[2] / self.anchors[best_anchor][0]),
                        np.log(box[3] / self.anchors[best_anchor][1]), box_class
                    ],
                    dtype=np.float32)
                matching_true_boxes[i, j, best_anchor] = adjusted_box
        return detectors_mask, matching_true_boxes

    def get_detector_mask(self, true_boxes):
        detectors_mask = [0 for i in range(len(true_boxes))]
        matching_true_boxes = [0 for i in range(len(true_boxes))]
        for i, box in enumerate(true_boxes):
            detectors_mask[i], matching_true_boxes[i] = self.preprocess_true_boxes(box)

        return np.array(detectors_mask), np.array(matching_true_boxes)

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]
        # 读取图片
        image = cv2.imread(image_path)
        # 获取图片原尺寸
        orig_size = np.array([image.shape[1], image.shape[0]])
        orig_size = np.expand_dims(orig_size, axis=0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
        # 将图片resize 到模型要求输入大小
        image = cv2.resize(image, self.input_shape)
        image = image / 255.

        boxes = np.array([np.array(box.split(","), dtype=np.int) for box in dataset[1:]])
        # 将真实像素坐标转换为(x_center, y_center, box_width, box_height, class)
        # 计算中点坐标
        boxes_xy = 0.5 * (boxes[:, 0:2] + boxes[:, 2:4])
        # 计算实际宽高
        boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]
        # 计算相对原尺寸的中点坐标和宽高
        boxes_xy = boxes_xy / orig_size
        boxes_wh = boxes_wh / orig_size
        # 拼接上面的x，y，w，h，c，为N*5矩阵, N为图像中实际的box数量
        boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 4:]), axis=1)
        # 填充boxes
        box_data = np.zeros((self.max_boxes, 5))
        if len(boxes) > self.max_boxes:
            boxes = boxes[:self.max_boxes]
        box_data[:len(boxes)] = boxes
        return image, box_data

    def data_generation(self, batch):
        """
        生成批量数据
        :param batch:
        :return:
        true_boxes:tensor，真实的boxes tensor shape[batch, num_true_boxes, 5]
            containing x_center, y_center, width, height, and class.
        detectors_mask: array detector 掩码，对于iou最大的anchor的位置为1，
            0/1 mask for detector positions where there is a matching ground truth.
        matching_true_boxes: array
            Corresponding ground truth boxes for positive detector positions.
            Already adjusted for conv height and width.
        y: 全零 [batch ]
        """
        images = []
        true_boxes = []
        for dataset in batch:
            image, box = self.read(dataset)
            images.append(image)
            true_boxes.append(box)
        images = np.array(images)
        # true_boxes B*N*5，B为一个批次图像的数量，N表示一个图像中允许的最大目标数量
        true_boxes = np.array(true_boxes)
        # 根据true_boxes 生成detectors_mask和matching_true_boxes
        detectors_mask, matching_true_boxes = self.get_detector_mask(true_boxes)

        return [images, true_boxes, detectors_mask, matching_true_boxes], np.zeros(self.batch_size)
