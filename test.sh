CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "SAM2UNet-Polyp.pth" \
--test_image_path "Kvasir-SEG/images/" \
--test_gt_path "Kvasir-SEG/masks/" \
--save_path "output_kvasir" \