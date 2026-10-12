"""Evaluate MSP-RANet checkpoints with the SAR-aware dataset pipeline."""

import argparse
from pathlib import Path


DEFAULT_DATA = Path(__file__).resolve().parent / "ultralytics/cfg/datasets/coco_ence_csar.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML (default: FAIR-CSAR).")
    parser.add_argument("--weights", required=True, help="Local MSP-RANet .pt checkpoint.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", default="0", help="GPU index (0), or cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001, help="Low cutoff for AP evaluation.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold, not the AP IoU threshold.")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="msp_ranet_test")
    parser.add_argument("--save-txt", action="store_true", help="Save predictions to labels/*.txt.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in saved labels.")
    parser.add_argument("--save-json", action="store_true", help="Save a prediction JSON file.")
    args = parser.parse_args()
    for field in ("data", "weights"):
        path = Path(getattr(args, field)).expanduser().resolve()
        if not path.is_file():
            parser.error(f"--{field} file does not exist: {path}")
        setattr(args, field, str(path))
    if Path(args.weights).suffix.lower() != ".pt":
        parser.error("--weights must point to a local .pt checkpoint.")
    if args.save_conf and not args.save_txt:
        parser.error("--save-conf requires --save-txt.")
    return args


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    options = vars(args).copy()
    options.pop("weights")
    metrics = model.val(**options, plots=True)
    print(f"Evaluation outputs: {metrics.save_dir}")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
