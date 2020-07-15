# yolov2-tensorflow2.2
# 环境
* 系统：win10
* tensorflow版本：2.2
* Anaconda3-2019.10
* python：3.7
* cuda：10.2
参考：[YAD2K](https://github.com/allanzelener/YAD2K)，其实就是将他的代码稍作修改
# 模型转换
执行命令 python convert.py config_path weights_path output_path  
例如：python convert.py model_data/yolov2.cfg model_data/yolov2.weights model_data/yolov2.h5  
config_path：yolov2的配置文件路径  
weights_path：yolov2的权重文件路径  
output_path：输出keras的h5文件  
# 生成训练数据文件
使用[voc2007数据集](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)，将其解压到和voc_annotation.py同级目录中  
执行命令 python voc_annotation.py  
在同级目录中会生成训练，测试，验证的数据文件  
# 训练
首先打开train.py 修改一些路径参数  
def _main():  
    # 训练数据文件路径  
    train_path = "2007_train.txt"  
    # 验证数据文件路径  
    val_path = "2007_val.txt"  
    # 日志文件路径  
    log_dir = 'logs/000/'  
    # voc classes name path  
    classes_path = "model_data/voc_classes.txt"  
    # anchors 文件路径  
    anchors_path = "model_data/yolov2_anchors.txt"  
    
确保model_data路径下有yolov2.h5文件  
如果GPU显存太小，可以将batch_size改为8，如果很大可以为64  
执行 python train.py   

# 测试  
修改yolo_test.py 文件  
def _main():  
    # TODO: 定义路径  
    # 训练后的权重文件  
    model_path = "model_data/yolov2_trained.h5"  
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'  
    # anchors 路径  
    anchors_path = "model_data/yolov2_anchors.txt"  
    # class name 路径  
    classes_path = "model_data/voc_classes.txt"  
    # 测试图片路径  
    test_path = "images/person.jpg"  
    # 输出图片路径  
    output_path = "images/person_out.jpg"  
    
执行命令：python yolo_test.py  
