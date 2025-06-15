import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from MAFCN.model.MA_FCN1 import NetFB
from MAFCN.model.Unet import Unet
from MAFCN.utils.tools import untransform_pred
import csv

def load_model(model_type='MAFCN'):
    if model_type == 'MAFCN':
        model = NetFB(class_num=2)
        encoder_weights = torch.load('C:\\Projects\\Processamento trab final\\MAFCN\\pretrained_model\\vgg16bn-encoder.pkl')
        model.features.load_state_dict(encoder_weights)
    else:
        model = Unet(num_classes=2)
        fcn_weights = torch.load('../MAFCN/pretrained_model/fcn8s_from_caffe.pth')
        new_params = model.state_dict().copy()
        for name, param in fcn_weights.items():
            if name in new_params:
                new_params[name] = param
        model.load_state_dict(new_params)

    model.eval()
    model.cuda()
    return model

def count_buildings(pred_mask):
    binary_mask = np.array(pred_mask).astype(bool)
    labeled, num_buildings = ndimage.label(binary_mask)
    return num_buildings

def load_image(image_path_base):
    try:
        return Image.open(image_path_base + '.png').convert('RGB')
    except Exception:
        try:
            return Image.open(image_path_base + '.tif').convert('RGB')
        except Exception as e:
            print(f"Failed to load image: {image_path_base}.png or .tif\nError: {e}")
            return None

def predict_buildings(model, image_path, model_type='MAFCN'):
    img = load_image(image_path)
    if img is None:
        return 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        if model_type == 'MAFCN':
            output, _, _, _, _ = model(img_tensor)
        else:
            output = model(img_tensor)

    pred_mask = untransform_pred(output[0]).convert('1')
    return count_buildings(pred_mask)

def compare_folders(before_folder, after_folder, model_type='MAFCN', output_csv='results.csv'):
    model = load_model(model_type)

    before_png = sorted(glob.glob(os.path.join(before_folder, '*.png')))
    before_tif = sorted(glob.glob(os.path.join(before_folder, '*.tif')))
    after_png = sorted(glob.glob(os.path.join(after_folder, '*.png')))
    after_tif = sorted(glob.glob(os.path.join(after_folder, '*.tif')))

    before_images = {os.path.splitext(os.path.basename(p))[0]: p for p in before_png + before_tif}
    after_images = {os.path.splitext(os.path.basename(p))[0]: p for p in after_png + after_tif}
    common_names = set(before_images.keys()) & set(after_images.keys())

    if not common_names:
        print("No matching filenames found between folders!")
        return

    results = []
    for name in common_names:
        before_path_base = os.path.join(before_folder, name)
        after_path_base = os.path.join(after_folder, name)

        before_count = predict_buildings(model, before_path_base, model_type)
        after_count = predict_buildings(model, after_path_base, model_type)
        change = after_count - before_count

        results.append({
            'filename': name,
            'before': before_count,
            'after': after_count,
            'change': change,
            'change_type': 'added' if change > 0 else 'removed' if change < 0 else 'unchanged'
        })

        print(f"{name}: Past={before_count}, Present={after_count}, Change={change}")

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'before', 'after', 'change', 'change_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")

if __name__ == '__main__':
    BEFORE_FOLDER = './past'
    AFTER_FOLDER = './present'
    compare_folders(BEFORE_FOLDER, AFTER_FOLDER, model_type='MAFCN')
