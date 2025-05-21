python3 main.py \
--model_mode train \
--model_name yolov8l.pt \
--epochs 60 \
--imgsz 640 \
--dataset_location datasets \
--hyp hyp.aml.baseline.yaml \
--freeze [16,17,18,19,20,21,22] \
