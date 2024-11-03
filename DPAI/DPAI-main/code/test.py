import os
import torch
import h5py
import numpy as np
import logging
from tqdm import tqdm
import nibabel as nib
from networks.net_factory import net_factory
from utils.test_3d_patch import test_single_case, calculate_metric_percase

# Parameter parsing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default="../data", help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='DPAI', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--split', type=str,  default='test', help='datalist to use')
args = parser.parse_args()

# Load the model
def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define the evaluation function
def var_all_case_self_train(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path="../predict"):
    with open(os.path.join(args.root_path, args.split + '_la.txt'), 'r') as f:
        image_list = f.readlines()
    image_list = [os.path.join(args.root_path, item.replace('\n', '').split(",")[0] + '.h5') for item in image_list]
    loader = tqdm(image_list)
    total_dice, total_jc, total_hd, total_asd = 0.0, 0.0, 0.0, 0.0
    num_cases = len(image_list)
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, _ = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        # Check if both prediction and label contain binary objects
        if np.sum(prediction) == 0 or np.sum(label) == 0:
            logging.warning("No binary object found in prediction or label for case: {}".format(image_path))
            num_cases -= 1
            continue  # Skip this case

        # Calculate metrics
        dice, jc, hd, asd = calculate_metric_percase(prediction, label)

        # Update total metrics
        total_dice += dice
        total_jc += jc
        total_hd += hd
        total_asd += asd

        # Save prediction results
        if save_result and test_save_path is not None:
            case_id = os.path.basename(image_path).replace('.h5', '')
            pred_file = os.path.join(test_save_path, case_id + "_pred.nii.gz")
            gt_file = os.path.join(test_save_path, case_id + "_gt.nii.gz")

            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), pred_file)
            nib.save(nib.Nifti1Image(label.astype(np.float32), np.eye(4)), gt_file)

    if num_cases > 0:
        avg_dice = total_dice / num_cases
        avg_jc = total_jc / num_cases
        avg_hd = total_hd / num_cases
        avg_asd = total_asd / num_cases
        print('Average Dice Coefficient: {:.4f}'.format(avg_dice))
        print('Average Jaccard Coefficient: {:.4f}'.format(avg_jc))
        print('Average Hausdorff Distance: {:.4f}'.format(avg_hd))
        print('Average Average Surface Distance: {:.4f}'.format(avg_asd))
    else:
        print('No valid cases found.')

    return avg_dice, avg_jc, avg_hd, avg_asd

if __name__ == "__main__":
    # Parameter settings
    model_path = "model.pth"
    model_type = args.model
    num_classes = 2  # assuming binary classification (background and foreground)
    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4

    # Load the model
    model = net_factory(net_type=model_type, in_chns=1, class_num=num_classes, mode="test")
    model = load_model(model_path, model)

    # Evaluate the model
    avg_dice, avg_jc, avg_hd, avg_asd = var_all_case_self_train(model, num_classes, patch_size, stride_xy, stride_z, save_result=True, test_save_path="../predict")

    print('Evaluation Results:')
    print('Average Dice Coefficient: {:.4f}'.format(avg_dice))
    print('Average Jaccard Coefficient: {:.4f}'.format(avg_jc))
    print('Average Hausdorff Distance: {:.4f}'.format(avg_hd))
    print('Average Average Surface Distance: {:.4f}'.format(avg_asd))
