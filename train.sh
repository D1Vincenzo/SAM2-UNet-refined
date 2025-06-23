CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "sam2_hiera_large.pt" \
--train_roots datasets/ClinicDB/train datasets/CVC-ColonDB/train datasets/ETIS/train datasets/Kvasir-SEG/train \
--save_path "." \
--epoch 20 \
--lr 0.001 \
--batch_size 2 \
--name "test" \
