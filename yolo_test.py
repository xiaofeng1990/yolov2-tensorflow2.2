import os
import random
import colorsys
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from yad2k.models.keras_yolo import yolo_eval, yolo_head, yolo_body


def get_classes(classes_path):
    """
    loads the classes
    :param classes_path: classes file path
    :return: list classes name
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """
    loads the anchors from a file
    :param anchors_path: anchors file path
    :return: array anchors shape:(5, 2)
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def _main():
    # TODO: 定义路径
    model_path = "model_data/yolov2_trained.h5"
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = "model_data/yolov2_anchors.txt"
    classes_path = "model_data/voc_classes.txt"
    test_path = "images/person.jpg"
    output_path = "images/person_out.jpg"
    # TODO: 判断文件是否存在

    # TODO: 加载anchors 和 classes_name
    anchors = get_anchors(anchors_path)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    # TODO: 加载模型
    image_input = Input(shape=(416, 416, 3))
    yolo_model = yolo_body(image_input, len(anchors), num_classes)
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
    final_layer = Conv2D(len(anchors) * (5 + num_classes), (1, 1), activation='linear')(topless_yolo.output)
    model = Model(image_input, final_layer)

    # TODO: 加载权重
    model.load_weights(model_path)

    # model.summary()
    # TODO: 设置方框颜色
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # TODO: 加载图片
    image = cv2.imread(test_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
    image = cv2.resize(image, (416, 416))
    image = image / 255
    image = np.expand_dims(image, 0)  # Add batch dimension.

    # TODO: 检测图片
    y = model.predict(image, batch_size=1)
    yolo_outputs = yolo_head(y, anchors, len(class_names))
    boxes, scores, classes = yolo_eval(yolo_outputs, (416, 416), score_threshold=0.6, iou_threshold=0.5)
    print('Found {} boxes for {}'.format(len(boxes), test_path))
    image = cv2.imread(test_path)
    origin_shape = image.shape[0:2]
    image = cv2.resize(image, (416, 416))
    for i, box in enumerate(boxes):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[classes[i]])
        cv2.putText(image, class_names[classes[i]], (int(box[0]), int(box[1])), 1, 1, colors[classes[i]], 1)

    image = cv2.resize(image, (origin_shape[1], origin_shape[0]))
    cv2.imshow('image', image)
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)


if __name__ == "__main__":
    _main()
