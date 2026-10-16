import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("run/fair-csar/v11/weights/best.pt")
    metrics = model.val(
        data="ultralytics/cfg/datasets/coco_ence.yaml",
        split="test",
        imgsz=1024,
        batch=10,
        workers=2,
        device="0",
        conf=0.2,
        iou=0.5,
        max_det=300,
        half=False,
        augment=False,
        plots=True,
        save=True,
        save_txt=True,
        project="run/fair-csar/v11",
        name="test",
    )
    print(metrics.results_dict)
