import warnings  # 导入warnings模块，用来控制Python中各种警告信息的显示行为

warnings.filterwarnings('ignore')  # 忽略所有的警告信息，避免在控制台输出干扰信息

from ultralytics import YOLO  # 从ultralytics包中导入YOLO类，用于模型加载、验证等操作

if __name__ == '__main__':  # 判断当前脚本是否作为主程序执行，若是则执行下面代码
    # 创建一个YOLO模型实例，这里直接加载训练好的best.pt权重文件
    model = YOLO('weight/fair-csar/v11/weights/best.pt')
    # 使用model.val()方法进行验证，传入各项验证参数
    model.val(
        data=r'ultralytics/cfg/datasets/coco_ence_csar.yaml',  # 数据集配置文件，其中包含训练集和验证集的路径、类别等信息
        split='test',  # 指定要使用的数据划分，这里选择验证集(val)
        imgsz=1024,  # 输入图像的尺寸
        batch=8,  # 验证时的批次大小
        project='runs',  # 保存验证结果（日志、输出文件等）的主目录
        name='test',  # 本次验证的名称，结果将保存在runs/val/exp目录下
        device='cuda:0',
        conf=0.2,
        save=True,  # 不保存带标注的图像
        save_txt=True, 
    )
