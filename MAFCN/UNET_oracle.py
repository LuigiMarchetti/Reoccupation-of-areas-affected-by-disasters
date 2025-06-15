import argparse
import torch
import torch.nn as nn
from torch.utils import data as Data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import os.path as osp
import time
from utils.tools import *
from dataset.datainput_all import InputData
from model.Unet import Unet

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_arguments():
    parser = argparse.ArgumentParser(description="UNET")

    # dataset
    parser.add_argument("--data_dir", type=str,
                        default='/media/disk2/lmy/smallall/train/',
                        help="dataset path.")
    parser.add_argument("--data_dir_val", type=str, default='/media/disk2/lmy/smallall/val/',
                        help="target dataset path.")
    parser.add_argument("--input_size", type=str, default='512,512',
                        help="width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    # network
    parser.add_argument("--batch_size", type=int, default=16,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="base learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_epoch", type=int, default=100,
                        help="number of training epochs.")
    parser.add_argument("--restore_from", type=str,
                        default='./pretrained_model/fcn8s_from_caffe.pth',
                        help="pretrained VGG model.")
    parser.add_argument("--weight_decay", type=float, default=0.00005,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")

    # result
    parser.add_argument("--snapshot_dir", type=str, default='/media/disk2/lmy/dg/Unet/smallall/Snap_UNET/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()

args = get_arguments()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter,num_steps):
    lr = lr_poly(args.learning_rate, i_iter, num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    if os.path.exists(args.snapshot_dir) == False:
        os.makedirs(args.snapshot_dir)

    pred_img_path = args.snapshot_dir + '/pred_img/'
    if os.path.exists(pred_img_path) == False:
        os.mkdir(pred_img_path)

    f = open(args.snapshot_dir + 'train.txt', 'w')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    # Create network
    net = Unet(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)

    new_params = net.state_dict().copy()
    for i, j in zip(saved_state_dict, new_params):
        if (i[0] != 'f') & (i[0] != 's') & (i[0] != 'u'):
            new_params[j] = saved_state_dict[i]

    net.load_state_dict(new_params)

    net = net.cuda()

    loader = Data.DataLoader(InputData(args.data_dir, crop_size=input_size),
                                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True)


    val_loader = Data.DataLoader(InputData(args.data_dir_val, crop_size=input_size),
                                 batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    num_batches = len(loader)

    optimizer = optim.Adam(net.parameters(),
                           lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    num_steps = args.num_epoch * num_batches
    loss_hist = np.zeros((num_steps, 5))
    index_i = -1
    OA_hist = 0.2

    tem_time_all = time.time()
    time_avg = 0.0
    for epoch in range(args.num_epoch):
        tem_time_epoch = time.time()
        for batch_index, data in enumerate(loader):
            index_i += 1

            tem_time = time.time()
            net.train()
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, index_i, num_steps)
            # train with source
            images, label, im_name = data
            images = images.cuda()
            label = label.cuda()
            # print(label.shape)
            output = net(images)
            # print(output.shape)
            # Segmentation Loss
            # cls_loss_value = loss_calc(output, label)
            cls_loss_value = loss_calc(output, label)
            # _, predict_labels = torch.max(output, 1)
            _, predict_labels = torch.max(output, 1)

            lbl_pred = predict_labels.detach().cpu().numpy()
            # print(np.unique(lbl_pred))
            lbl_true = label.detach().cpu().numpy()
            # print(np.unique(lbl_true))
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _,acc_cls,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch.append((acc_cls,mean_iu))
            miu = np.mean(metrics_batch, axis=0)

            total_loss = cls_loss_value
            total_loss.backward()

            loss_hist[index_i, 0] = total_loss.item()
            loss_hist[index_i, 1] = cls_loss_value.item()
            loss_hist[index_i, 2] = miu[1]
            loss_hist[index_i, 3] = miu[0]

            optimizer.step()
            batch_time = time.time() - tem_time

            if (batch_index + 1) % 10 == 0:
                print('epoch %d/%d:  %d/%d time: %.6f miu = %.2f acc = %.2f cls_loss = %.6f\n' % (
                epoch + 1, args.num_epoch, batch_index + 1, num_batches, batch_time,
                np.mean(loss_hist[index_i - 9:index_i + 1, 2]) * 100,np.mean(loss_hist[index_i - 9:index_i + 1, 3]) * 100,
                np.mean(loss_hist[index_i - 9:index_i + 1, 1])))
                f.write('epoch %d/%d:  %d/%d time: %.2f miu = %.2f acc = %.2f cls_loss = %.6f\n' % (
                epoch + 1, args.num_epoch, batch_index + 1, num_batches, batch_time,
                np.mean(loss_hist[index_i - 9:index_i + 1, 2]) * 100,np.mean(loss_hist[index_i - 9:index_i + 1, 3]) * 100,
                np.mean(loss_hist[index_i - 9:index_i + 1, 1])))
                f.flush()

            if (index_i + 1 ) % 1000 == 0:
                pred = output[0, :, :, :]
                # print(pred.shape)
                true = label[0, :, :]
                pred = untransform_pred(pred)
                true = untransform_label(true)
                pred.save('%s/iter%012d_%s_pred.tif' % (pred_img_path,index_i + 1,im_name[0]))
                true.save('%s/iter%012d_%s_true.tif' % (pred_img_path,index_i + 1,im_name[0]))


        epoch_time = time.time()-tem_time_epoch
        time_avg = (time_avg * (epoch) + epoch_time) / (epoch + 1)
        print('epoch_time %.6f  epoch_avg_time %.6f\n' % (epoch_time, time_avg))
        f.write('epoch_time %.6f  epoch_avg_time %.6f\n' % (epoch_time, time_avg))

        OA_new = test_mIoU(f, net, val_loader, epoch + 1,n_class=args.num_classes)

        # Saving the models
        if OA_new > OA_hist:
            f.write('Save Model\n')
            print('Save Model')
            model_name = 'epoch' + repr(epoch + 1) + 'miu_' + repr(
                int(OA_new * 1000)) + '.pth'
            torch.save(net.state_dict(), os.path.join(
                args.snapshot_dir, model_name))
            OA_hist = OA_new

    all_time = time.time() - tem_time_all
    print('all_time %.6f\n' % all_time)
    f.write('all_time %.6f\n' % all_time)
    f.close()
    torch.save(net.state_dict(), os.path.join(
        args.snapshot_dir, 'Net.pth'))
    np.savez(args.snapshot_dir + 'loss.npz', loss_hist=loss_hist)


if __name__ =='__main__':
    main()




