python3 main.py \
--model_mode val \
--model_name runs/detect/train12/weights/best.pt \ #update this 
--epochs 5 \
--batch 16 \
--imgsz 640 \
--dataset_location datasets \
--iou 0.5 \
--conf 0.25 \
