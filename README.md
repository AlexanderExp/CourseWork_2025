# 🚜 PPE-Detection

Репозиторий для **детекции техники, людей** на аэро-снимках полей (VisDrone + Tractor-4).  
Код полностью воспроизводим — используем **Git + DVC, TensorBoard, локальный форк Ultralytics**.

## Быстрый старт

#### установка
git clone <repo>
cd PPE-Detection
pip install -r requirements.txt          # PyTorch-CUDA, DVC, TensorBoard…
pip install -e .                         # локальный форк ultralytics (editable)

#### запуск baseline-обучения
dvc exp run                              # использует configs/params_baseline.yaml

#### лог-кривые
tensorboard --logdir tensorboard_logs
