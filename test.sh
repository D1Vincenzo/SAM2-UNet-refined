CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "SAM2-UNet-200-medsam-labels-lr001.pth" \
--test_image_path "processed_dataset_labels/test/images/" \
--test_gt_path "processed_dataset_labels/test/masks/" \
--save_path "outputs_medsam_200_labels-lr001/" \
--target_labels 0 6 7 8 9