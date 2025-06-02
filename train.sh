CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "sam2_hiera_large.pt" \
--train_image_path "Kvasir-SEG/images/" \
--train_mask_path "Kvasir-SEG/masks/" \
--save_path "." \
--epoch 20 \
--lr 0.001 \
--batch_size 1