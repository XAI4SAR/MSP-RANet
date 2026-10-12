"""Train MSP-RANet using the customized Ultralytics source in this repository."""

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/cfg/models/11/yolo11-obb-rapconv.yaml"
DEFAULT_DATA = PROJECT_ROOT / "ultralytics/cfg/datasets/coco_ence_csar.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML (default: FAIR-CSAR).")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local model YAML or .pt checkpoint.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0", help="GPU index (0), or cpu.")
    parser.add_argument("--workers", type=int, default=4, help="Use 0 if multiprocessing causes problems.")
    parser.add_argument("--optimizer", default="SGD", choices=("SGD", "Adam", "AdamW", "auto"))
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="msp_ranet_train")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision; disabled by default.")
    parser.add_argument("--cache", action="store_true", help="Cache training images in RAM.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted --model last.pt run.")
    # The current augmentation pipeline does not preserve priors through Mosaic/HSV.
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.0)
    args = parser.parse_args()
    for field in ("data", "model"):
        path = Path(getattr(args, field)).expanduser().resolve()
        if not path.is_file():
            parser.error(f"--{field} file does not exist: {path}")
        setattr(args, field, str(path))
    if args.resume and Path(args.model).suffix.lower() != ".pt":
        parser.error("--resume requires --model pointing to an interrupted run's last.pt.")
    return args


def main() -> None:
    args = parse_args()
    # Keep --help usable before installing the training dependencies.
    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.resume:
        # The backend restores the checkpoint's optimizer and training settings.
        model.train(resume=True, data=args.data, device=args.device)
    else:
        options = vars(args).copy()
        options.pop("model")
        options.update(pretrained=False, verbose=True, plots=True)
        model.train(**options)
    print(f"Training outputs: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
