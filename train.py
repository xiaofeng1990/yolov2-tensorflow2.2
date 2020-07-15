import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yad2k.models.keras_yolo import yolo_body, yolo_loss
from yad2k.utils.yolo_loss import YoloLoss
from yad2k.utils.data_sequence import SequenceData


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


def create_model(anchors, num_classes, load_pretrained=True, freeze_body=True, weights_path="model_data"):
    """
    创建 yolov2 模型
    :param anchors: anchors 数目
    :param num_classes: 类别数目
    :param load_pretrained: 加载权重标志，默认为True，加载权重
    :param freeze_body: 是否冻结cnn
    :param weights_path: 权重路径
    :return: yolo 模型
    model_body: YOLOv2 with new output layer
    model: YOLOv2 with custom loss Lambda layer
    """
    # TODO: 1.检查weights_path 目录是否存在，不存在就创建
    # TODO: 2.创建模型
    # Create model input layers.
    # 13*13为单元格矩阵，5表示每个单元格有5个anchor，第三维度表示如果该anchor有目标就为1，否则为0
    detectors_mask_shape = (13, 13, 5, 1)
    # 最后一个5表示（x, y, w, h, c）c表示类别索引范围是0--1，目标放到了对应的单元格中的anchor
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    # 输入图片为416*416，三通道（RGB）
    image_input = Input(shape=(416, 416, 3))
    # None表示一个图片中的目标可以不确定，boxes_input表示一张图中所有目标信息，可以设置最大值比如20，表示训练数据中一张图片允许最多有20个目标
    boxes_input = Input(shape=(None, 5))
    # 目标掩码，确定目标位于哪一个单元格中的哪一个anchor
    detectors_mask_input = Input(shape=detectors_mask_shape)
    # 目标在单元格中的anchor的编码位置和类别信息
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), num_classes)
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    # TODO: 3.加载权重
    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolov2.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    # 冻结yolo body，只训练最后一层
    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    # 根据anchor的数量和classes的数量，来创建最后一个卷积层
    final_layer = Conv2D(len(anchors) * (5 + num_classes), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # loss 创建损失层
    model_loss = YoloLoss(anchors, num_classes)(
        [model_body.output, boxes_input, detectors_mask_input, matching_boxes_input])
    """

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={'anchors': anchors,
                   'num_classes': num_classes})([
        model_body.output, boxes_input,
        detectors_mask_input, matching_boxes_input
    ])
    """
    model = Model(
        [image_input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model


def _main():
    train_path = "2007_train.txt"
    val_path = "2007_val.txt"
    log_dir = 'logs/000/'
    classes_path = "model_data/voc_classes.txt"
    anchors_path = "model_data/yolov2_anchors.txt"
    # TODO: 加载class_name
    class_name = get_classes(classes_path)
    num_classes = len(class_name)
    # TODO: 加载anchors
    anchors = get_anchors(anchors_path)
    # 输入图片size 必须是32的整数倍
    input_shape = (416, 416)
    # 训练批次大小，这里可以根据你的GPU显存更改，如果显存很大，可以改为32或者64
    batch_size = 8
    epochs = 100

    # 初始化GPU，内存分配用多少分配多少
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # TODO: 创建模型
    model_body, model = create_model(anchors, num_classes)
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.
    # 打印模型
    model.summary()
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 创建数据发生次
    train_sequence = SequenceData(train_path, input_shape, batch_size, anchors, num_classes)
    val_sequence = SequenceData(val_path, input_shape, batch_size, anchors, num_classes)

    model.fit_generator(train_sequence,
                        steps_per_epoch=train_sequence.get_epochs(),
                        validation_data=val_sequence,
                        validation_steps=val_sequence.get_epochs(),
                        epochs=epochs,
                        workers=4,
                        callbacks=[checkpoint, early_stopping])
    model.save_weights("model_data/yolov2_trained.h5")
    print("over****************")


if __name__ == "__main__":
    _main()
