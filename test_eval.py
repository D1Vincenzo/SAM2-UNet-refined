import os
import argparse
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import cv2
import py_sod_metrics
from SAM2UNet import SAM2UNet
from dataset import TestDataset


def evaluate_predictions(pred_root, mask_root, dataset_name):
    FM = py_sod_metrics.Fmeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()

    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "dice": py_sod_metrics.DICEHandler(**sample_gray),
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
        }
    )

    mask_name_list = sorted(os.listdir(mask_root))
    for i, mask_name in enumerate(mask_name_list):
        print(f"[{i}] Processing {mask_name}...")
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name[:-4] + '.png')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if pred is None:
            print(f"[WARN] Cannot read prediction image: {pred_path}")
            continue
        if mask is None:
            print(f"[WARN] Cannot read GT mask image: {mask_path}")
            continue

        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)
        FMv2.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]
    fmv2 = FMv2.get_results()

    curr_results = {
        "meandice": fmv2["dice"]["dynamic"].mean(),
        "meaniou": fmv2["iou"]["dynamic"].mean(),
        'Smeasure': sm,
        "wFmeasure": wfm,
        "adpFm": fm["adp"],
        "meanEm": em["curve"].mean(),
        "MAE": mae,
    }

    print(dataset_name)
    print("mDice:       ", format(curr_results['meandice'], '.3f'))
    print("mIoU:        ", format(curr_results['meaniou'], '.3f'))
    print("S_{alpha}:   ", format(curr_results['Smeasure'], '.3f'))
    print("F^{w}_{beta}:", format(curr_results['wFmeasure'], '.3f'))
    print("F_{beta}:    ", format(curr_results['adpFm'], '.3f'))
    print("E_{phi}:     ", format(curr_results['meanEm'], '.3f'))
    print("MAE:         ", format(curr_results['MAE'], '.3f'))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)

    model = SAM2UNet(n_experts=args.n_experts, top_k=args.top_k).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    model.cuda()
    os.makedirs(args.save_path, exist_ok=True)

    for i in range(test_loader.size):
        with torch.no_grad():
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.to(device)

            res, _, _ = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            # Optional binarization
            lambda_ = 0.5
            res[res >= int(255 * lambda_)] = 255
            res[res < int(255 * lambda_)] = 0

            save_path = os.path.join(args.save_path, name[:-4] + ".png")
            imageio.imsave(save_path, res)
            print(f"Saved prediction: {save_path}")

    # After all predictions
    evaluate_predictions(args.save_path, args.test_gt_path, args.dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to the checkpoint of sam2-unet")
    parser.add_argument("--test_image_path", type=str, required=True,
                        help="path to the image files for testing")
    parser.add_argument("--test_gt_path", type=str, required=True,
                        help="path to the mask files for testing")
    parser.add_argument("--save_path", type=str, required=True,
                        help="path to save the predicted masks")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="name of the test dataset (used in print only)")
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)

    args = parser.parse_args()
    main(args)
