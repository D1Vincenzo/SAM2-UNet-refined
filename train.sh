CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "MedSAM2_latest.pt" \
--train_image_path "processed_dataset_labels/train/images/" \
--train_mask_path "processed_dataset_labels/train/masks/" \
--save_path "." \
--epoch 20 \
--lr 0.001 \
--batch_size 2 \
--target_labels 0 6 7 8 9