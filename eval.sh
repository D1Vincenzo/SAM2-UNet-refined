# python eval.py \
# --dataset_name "syn_adapter" \
# --pred_path "outputs_adapter" \
# --gt_path "processed_dataset/test/masks" \

# python eval.py \
# --dataset_name "syn_lora" \
# --pred_path "outputs_lora" \
# --gt_path "processed_dataset/test/masks" \

python eval.py \
--dataset_name "syn_adapter_small" \
--pred_path "outputs_adapter_small" \
--gt_path "processed_dataset/test/masks_small" \

python eval.py \
--dataset_name "syn_lora_small" \
--pred_path "outputs_lora_small" \
--gt_path "processed_dataset/test/masks_small" \

# python eval.py \
# --dataset_name "syn_lora_200" \
# --pred_path "outputs_lora_200" \
# --gt_path "processed_dataset/test/masks" \

python eval.py \
--dataset_name "syn_lora_200_small" \
--pred_path "outputs_lora_200_small" \
--gt_path "processed_dataset/test/masks_small" \
