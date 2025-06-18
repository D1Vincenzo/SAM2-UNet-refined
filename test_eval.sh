python test_eval.py \
  --checkpoint moe-adapter8-notopk-best-model.pth \
  --test_image_path datasets/ETIS/test/images/ \
  --test_gt_path datasets/ETIS/test/masks/ \
  --dataset_name adapter8-notopk_ETIS \
  --n_experts 8 \
  --top_k 8 \
  --save_path OP_test_ETIS \

