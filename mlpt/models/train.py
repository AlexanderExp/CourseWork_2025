import time
import json
import yaml
import argparse
import os
import sys
from mlpt.utils.utils import train_and_validate_models
from mlpt.utils.tb_logger import TensorBoardLogger

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")))
os.environ["MPLBACKEND"] = "agg"


def load_params(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    params = load_params(config_path)
    data_cfg = params["data"]["config"]
    models = {params["model"]["name"]: params["model"]["weights"]}

    tcfg = params.get("training", {})
    epochs = tcfg.get("epochs", 1)
    batch = tcfg.get("batch", 1)
    imgsz = tcfg.get("imgsz", 320)
    mosaic = tcfg.get("mosaic", 0.0)
    mixup = tcfg.get("mixup", 0.0)
    augment = tcfg.get("augment", False)
    fraction = tcfg.get("fraction", 1.0)

    project = "runs/detect"
    print(f"[INFO] Training with epochs={epochs}, batch={batch}, imgsz={imgsz}")

    start = time.time()
    results = train_and_validate_models(
        models, data_cfg, project,
        epochs=epochs, batch=batch,
        imgsz=imgsz, mosaic=mosaic,
        mixup=mixup, augment=augment,
        fraction=fraction
    )
    tot_time = time.time() - start
    print(f"[INFO] Total training time: {tot_time:.2f} s")

    tb = TensorBoardLogger(base_log_dir="runs/detect/tensorboard")
    for r in results:
        tb.log_metrics(r["Model"], {
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "mAP50": r["mAP50"],
            "mAP50-95": r["mAP50-95"],
            "Train Time": r["Training Time (s)"]
        }, step=epochs)
    tb.close()

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Metrics written to metrics.json")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="params.yaml")
    args = p.parse_args()
    main(args.config)
