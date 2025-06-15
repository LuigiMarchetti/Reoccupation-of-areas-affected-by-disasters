import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import glob
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.MA_FCN1 import NetFB
from dataset.datainput_all import InputData


from collections import OrderedDict
import os
from PIL import Image
from utils.tools import *

import matplotlib.pyplot as plt
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unet")
    parser.add_argument("--data_dir", type=str, default='/media/disk2/lmy/dataset/smallall/test1/downtown/',
                        help="target dataset path/media/disk2/lmy/dataset/dunedin/media/disk2/lmy/dataset/potsdam.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='/media/disk2/lmy/dg/MAFCN/whu_set/Snap_MAFCN1/',
                        help="restored model.")
    parser.add_argument("--snapshot_dir", type=str, default='/media/disk2/lmy/dg/MAFCN/whu_set/Snap_MAFCN1/Maps_downtown/',
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    f = open(args.snapshot_dir + 'Evaluation.txt', 'w')

    model = NetFB(class_num=args.num_classes)

    testloader = data.DataLoader(InputData(args.data_dir, crop_size=(512, 512)), batch_size=1,
                                 shuffle=False,pin_memory=True)

    input_size = (512, 512)
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    models_path = glob.glob(args.restore_from + 'epoch28miu_929.pth')
    for model_path in models_path:
        print('\n evaluation for :%s' % (model_path) + '\n')
        f.write('\n evaluation for :%s' % (model_path) + '\n')

        saved_state_dict = torch.load(model_path)
        model.load_state_dict(saved_state_dict)

        save_name = model_path.split('/')[-1].split('.')[0]
        save_path = args.snapshot_dir + save_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model.eval()
        model.cuda()

        test_mIoU_MAFCN(f, model, testloader, 0, n_class=args.num_classes)

        for index, batch in enumerate(testloader):
            if index % 1000 == 0:
                print('%d processd' % index)
            image, _, name = batch
            output,_,_,_,_ = model(image.cuda())
            # output = interp(output)
            output = output.cpu().data[0].numpy()

            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2)*255.0, dtype=np.uint8)

            output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            output.save('%s/%s' % (save_path, name))

    f.close()


if __name__ == '__main__':
    main()
