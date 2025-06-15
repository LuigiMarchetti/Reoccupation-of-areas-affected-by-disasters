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

def load_model(model_type='MAFCN'):
    """Load pretrained model with correct weights"""
    if model_type == 'MAFCN':
        model = NetFB(class_num=2)
        # Load VGG16 encoder weights
        encoder_weights = torch.load('../MAFCN/pretrained_model/vgg16bn-encoder.pkl')
        model.features.load_state_dict(encoder_weights)
    else:  # UNet
        model = Unet(num_classes=2)
        # Load FCN weights
        fcn_weights = torch.load('../MAFCN/pretrained_model/fcn8s_from_caffe.pth')
        # Adapt FCN weights to UNet architecture
        new_params = model.state_dict().copy()
        for name, param in fcn_weights.items():
            if name in new_params:
                new_params[name] = param
        model.load_state_dict(new_params)

    model.eval()
    model.cuda()
    return model

def count_buildings(pred_mask):
    """Count buildings using connected components"""
    # Convert to binary numpy array
    binary_mask = np.array(pred_mask).astype(bool)

    # Label connected components
    labeled, num_buildings = ndimage.label(binary_mask)
    return num_buildings

def predict_buildings(model, image_path, model_type='MAFCN'):
    """Predict building count from single image"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()

    # Get prediction
    with torch.no_grad():
        if model_type == 'MAFCN':
            output, _, _, _, _ = model(img_tensor)
        else:
            output = model(img_tensor)

    # Convert to binary mask (1=building, 0=background)
    pred_mask = untransform_pred(output[0]).convert('1')

    # Count actual buildings (connected components)
    return count_buildings(pred_mask)

def compare_folders(before_folder, after_folder, model_type='MAFCN', output_csv='results.csv'):
    """Compare building counts between before and after images"""
    # Load model
    model = load_model(model_type)

    # Get matching filenames
    before_images = sorted(glob.glob(os.path.join(before_folder, '*.png')))
    after_images = sorted(glob.glob(os.path.join(after_folder, '*.png')))

    # Find common images
    common_names = set(os.path.basename(f) for f in before_images) & \
                   set(os.path.basename(f) for f in after_images)

    if not common_names:
        print("No matching filenames found between folders!")
        return

    results = []
    for name in common_names:
        before_path = os.path.join(before_folder, name)
        after_path = os.path.join(after_folder, name)

        # Get building counts
        before_count = predict_buildings(model, before_path, model_type)
        after_count = predict_buildings(model, after_path, model_type)

        # Calculate change
        change = after_count - before_count

        results.append({
            'filename': name,
            'before': before_count,
            'after': after_count,
            'change': change,
            'change_type': 'added' if change > 0 else 'removed' if change < 0 else 'unchanged'
        })

        print(f"{name}: Buildings Before={before_count}, After={after_count}, Change={change:+d}")

    # Save results
    import csv
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'before', 'after', 'change', 'change_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")

if __name__ == '__main__':
    # Configuration
    BEFORE_FOLDER = './past'      # Folder with pre-earthquake images
    AFTER_FOLDER = './present'    # Folder with post-reconstruction images

    # Choose either 'MAFCN' or 'UNET' based on which model you want to use
    compare_folders(BEFORE_FOLDER, AFTER_FOLDER, model_type='MAFCN')