import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from ..models.keras_yolo import yolo_loss


class YoloLoss(Layer):
    def __init__(self, anchors, num_classes, **kwargs):
        super(YoloLoss, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes
        self._name = "yolo_loss"

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)

    def call(self, inputs, **kwargs):
        loss = yolo_loss(inputs, self.anchors, self.num_classes)
        self.add_loss(loss, inputs=True)
        self.add_metric(loss, aggregation="mean", name="yolo_loss")
        return loss

    def get_config(self):
        pass
