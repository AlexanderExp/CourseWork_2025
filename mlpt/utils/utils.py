import os
import time
import glob
import pandas as pd
import torch
from mlpt.modules.ultralytics.ultralytics import YOLO


def update(file_path: str, root: str) -> str:
    """
    Обновляет пути в текстовом файле, добавляя базовую директорию root.
    """
    with open(file_path, "r") as file:
        paths = file.readlines()
    updated_paths = [os.path.join(root, p.strip()) + "\n" for p in paths]
    new_file_path = os.path.join(
        os.path.dirname(file_path),
        "updated_" + os.path.basename(file_path)
    )
    with open(new_file_path, "w") as file:
        file.writelines(updated_paths)
    print("Updated file created successfully:", new_file_path)
    return new_file_path


def wait_for_results_file(run_folder: str, pattern: str = "results.csv",
                          timeout: int = 10) -> str:
    """
    Ждёт появления CSV-файла с результатами в папке run_folder.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        hits = glob.glob(os.path.join(run_folder, pattern))
        if hits:
            return hits[0]
        time.sleep(1)
    return ""


def train_and_validate_models(models_to_train: dict, data_config: str,
                              project_name: str, epochs: int = 10,
                              batch: int = 1, imgsz: int = 320,
                              mosaic: float = 0.0, mixup: float = 0.0,
                              augment: bool = False, fraction: float = 1.0) -> list:
    """
    Обучает и валидирует модели, возвращает список словарей с метриками.
    """
    # Создаём базовую папку для всех экспериментов
    os.makedirs(project_name, exist_ok=True)

    results = []
    for model_name, weights_path in models_to_train.items():
        print(f"\n=== Обучение модели {model_name} ===")
        model = YOLO(weights_path)
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        run_name = f"train_{model_name}"
        run_folder = os.path.join(project_name, run_name)
        # ГАРАНТИРУЕМ, что папка существует до model.train
        os.makedirs(run_folder, exist_ok=True)

        print(f"Начало тренировки -> project={project_name}, name={run_name}")
        start = time.time()
        model.train(
            data=data_config,
            epochs=epochs,
            project=project_name,
            name=run_name,
            exist_ok=True,       # чтобы перезаписать, если нужно
            batch=batch,
            imgsz=imgsz,
            mosaic=mosaic,
            mixup=mixup,
            augment=augment,
            fraction=fraction,
            verbose=True,
        )
        train_time = time.time() - start
        print(f"Тренировка завершена за {train_time:.2f} секунды.")

        # Валидация
        val = model.val(
            data=data_config,
            project=project_name,
            name=run_name,
            verbose=True
        )

        # Читаем CSV с метриками
        csv_path = wait_for_results_file(run_folder)
        if csv_path:
            df = pd.read_csv(csv_path).iloc[-1]
            precision = df.get("metrics/precision(B)", 0)
            recall = df.get("metrics/recall(B)", 0)
            mAP50 = df.get("metrics/mAP50(B)", 0)
            mAP50_95 = df.get("metrics/mAP50-95(B)", 0)
        else:
            precision = getattr(val, "precision", 0)
            recall = getattr(val, "recall", 0)
            mAP50 = getattr(val, "mAP50", 0)
            mAP50_95 = getattr(val, "mAP50_95", 0)

        print(
            f"Результаты {model_name}: "
            f"P={precision:.4f}, R={recall:.4f}, "
            f"mAP50={mAP50:.4f}, mAP50-95={mAP50_95:.4f}"
        )

        results.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
            "Training Time (s)": train_time,
        })

    return results



def aggregate_results(result_folders: list, output_csv_path: str) -> pd.DataFrame:
    """
    Агрегирует результаты из указанных папок, формирует сводную таблицу и сохраняет её в CSV.

    :param result_folders: Список путей к папкам с результатами.
    :param output_csv_path: Путь для сохранения итогового CSV-файла.
    :return: Сформированный DataFrame с итоговыми метриками.
    """
    results_list = []

    for run_folder in result_folders:
        model_name = os.path.basename(run_folder)
        results_file = wait_for_results_file(run_folder)

        if results_file:
            df_metrics = pd.read_csv(results_file)
            last_row = df_metrics.iloc[-1]
            precision = last_row.get("metrics/precision(B)", 0)
            recall = last_row.get("metrics/recall(B)", 0)
            mAP50 = last_row.get("metrics/mAP50(B)", 0)
            mAP50_95 = last_row.get("metrics/mAP50-95(B)", 0)
        else:
            precision = recall = mAP50 = mAP50_95 = 0

        results_list.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
        })

    results_df = pd.DataFrame(results_list)
    for col in ["Precision", "Recall", "mAP50", "mAP50-95"]:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce").fillna(0)

    results_df.to_csv(output_csv_path, index=False)
    return results_df


def plot_results(results_df: pd.DataFrame) -> None:
    """
    Строит графики для визуального сравнения метрик.

    :param results_df: DataFrame с результатами.
    """
    import matplotlib.pyplot as plt

    num_models = len(results_df)
    model_colors = plt.cm.tab10(range(num_models))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].bar(results_df["Model"], results_df["Precision"], color=model_colors)
    axes[0].set_title("Precision")
    axes[0].set_xlabel("Модель")
    axes[0].set_ylabel("Precision")

    axes[1].bar(results_df["Model"], results_df["Recall"], color=model_colors)
    axes[1].set_title("Recall")
    axes[1].set_xlabel("Модель")
    axes[1].set_ylabel("Recall")

    axes[2].bar(results_df["Model"], results_df["mAP50"], color=model_colors)
    axes[2].set_title("mAP50")
    axes[2].set_xlabel("Модель")
    axes[2].set_ylabel("mAP50")

    axes[3].bar(results_df["Model"], results_df["mAP50-95"], color=model_colors)
    axes[3].set_title("mAP50-95")
    axes[3].set_xlabel("Модель")
    axes[3].set_ylabel("mAP50-95")

    plt.tight_layout()
    plt.show(block=True)