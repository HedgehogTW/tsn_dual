import os
import pathlib
import time, datetime
import argparse
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import video_transforms
import models
import datasets
    
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# print('support model:', model_names)

dataset_names = sorted(name for name in datasets.__all__)
# print('dataset_names', dataset_names)

# print('datasets.__dict__', datasets.__dict__)
# ['ResNet', 'rgb_resnet18', 'rgb_resnet34', 'rgb_resnet50', 'rgb_resnet50_aux', 'rgb_resnet101',
           # 'rgb_resnet152']

# for evaluation: $ python main_twostream.py E:/tsn_data -e
# for training: $ python main_twostream.py e:/tsn_data 
parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('data', metavar='DIR', 
                    help='path to dataset')
# parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
#                     help='path to datset setting files')
# parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
#                     choices=["rgb", "flow"],
#                     help='modality: rgb | flow')
parser.add_argument('--dataset', '-d', default='rat_two',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='twostream_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_vgg16)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter_size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
# parser.add_argument('--new_length', default=1, type=int,
#                     metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=224, type=int,
                    metavar='N', help='resize width (default: 180)')
parser.add_argument('--new_height', default=224, type=int,
                    metavar='N', help='resize height (default: 240)')
# parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
#                     metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[80, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--segments', default=1, type=int,
                    metavar='N', help='number of segments (default: 1)')
parser.add_argument('--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
# parser.add_argument('--save-freq', default=20, type=int,
                    # metavar='N', help='save frequency (default: 25)')
# parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--trainall', dest='trainall', action='store_true',
                    help='train all layers')
parser.add_argument('--opti', '-o', default='adam',
                    choices=["sgd", "adam"],
                    help='optimizer: sgd | adam')

best_prec1 = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.evaluate = True

    rgb_length = 3
    flow_length = 10
    rgb_is_color = False
    flow_is_color = False
    args.modality = 'two'
    # args.arch = 'twostream_resnet50'

    # print('support models:', model_names)
    # create model
    
    print('new_length for rgb is {}, optical flow is {}'.format(rgb_length, flow_length))
    print('--evaluate ', args.evaluate)

    if rgb_is_color:
        rgb_channel = 3 * rgb_length * args.segments
    else:
        rgb_channel = rgb_length * args.segments

    flow_channel = flow_length * 2 * args.segments


    print("Model {} is loaded. rgb_channel {} color {}, flow_channel {}".
        format(args.arch, rgb_channel, rgb_is_color, flow_channel))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    args.resume = os.path.join(args.data, 'checkpoints_2s')
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("checkpoints directory %s." % (args.resume))

    cudnn.benchmark = True
       
    # Data transforming

    rgb_scale_ratios = [1.0, 0.875, 0.75, 0.66]
    rgb_clip_mean = [0.485, 0.456, 0.406] * rgb_length
    rgb_clip_std = [0.229, 0.224, 0.225] * rgb_length

    
    flow_scale_ratios = [1.0, 0.875, 0.75]
    flow_clip_mean = [0.5, 0.5] * flow_length
    flow_clip_std = [0.226, 0.226] * flow_length

    # rgb, flow 都用rgb 參數做normalize.
    normalize = video_transforms.Normalize(mean=rgb_clip_mean,
                                           std=rgb_clip_std)

    val_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            # video_transforms.CenterCrop((224)),
            video_transforms.ToTensor(),
            normalize,
        ])

    # data loading, train_rgb, train_flow 內容一樣

    test_setting_file = "test_split%d.txt" % (args.split)
    test_split_file = os.path.join(args.data, args.dataset, test_setting_file)

    if not os.path.exists(test_split_file): 
        print('no test file ', test_split_file)


    test_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                  source=test_split_file,
                                                  phase="val",
                                                  modality=args.modality,
                                                  is_color=(rgb_is_color, flow_is_color),
                                                  num_segments = args.segments,
                                                  new_length=(rgb_length, flow_length),
                                                  new_width=args.new_width,
                                                  new_height=args.new_height,
                                                  video_transform=val_transform)

    print('{} test samples found in {}'.format(len(test_dataset), test_split_file))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ################################ evaluate
    path_checkpoint = pathlib.Path(args.resume )
        
    chk_lst = sorted(path_checkpoint.glob('model_{}*.tar'.format(args.arch)))
    model_name = chk_lst[-1]
    print('model_name :', model_name )

    if os.path.isfile(model_name):
        print("==> loading model '{}'".format(model_name))
        model = torch.load(model_name)
    else:
        print("==> no model found at '{}'".format(model_name)) 

    prec1, df = validate(test_loader, model, criterion)        
    print(classification_report(df['target'], df['pred']))
    conf_mx = confusion_matrix(df['target'], df['pred'])
    print(conf_mx)

    opti = model_name.stem.split('_')[3]
    dt = datetime.datetime.now().strftime("%m%d_%H%M")    
    out_name = 'test_{}_{}_{}.csv'.format(args.arch, opti, dt)
    df.to_csv(out_name,  sep = ',')
    print('acc: {:.4f}, out file: {}'.format(prec1, out_name))
    return



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_a = AverageMeter()
    top1_f = AverageMeter()
    top1_final = AverageMeter()

    # switch to evaluate mode
    model.eval()

    clip_name_lst = []
    target_lst = []
    predict_lst = []
    end = time.time()
    for i, (input_a, input_b, target, index, path) in enumerate(val_loader):
        # print(index, path, target)
        input_var_a = input_a.float().cuda(non_blocking=True)
        input_var_b = input_b.float().cuda(non_blocking=True)
        
        target_var = target.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        with torch.no_grad():
            # compute output
            output_a, output_f = model(input_var_a, input_var_b)
            # loss = criterion(output, target_var)
            loss_a = criterion(output_a, target_var)
            loss_f = criterion(output_f, target_var)
            loss =0.4*loss_a + 0.6*loss_f


        # measure accuracy and record loss
        # prec1, pred = accuracy(output.data, target_var) #, topk=(1, 3))
        losses.update(loss.data, input_a.size(0))
        
        # top3.update(prec3, input.size(0))

        prec1_f, _ = accuracy(output_f.data, target_var) #, topk=(1, 3))
        prec1_a, _ = accuracy(output_a.data, target_var) #, topk=(1, 3))
        prec1_final, pred = accuracy_fusion(output_a.data, output_f.data, target_var)
        top1_a.update(prec1_a[0], input_a.size(0))
        top1_f.update(prec1_f[0], input_a.size(0))
        top1_final.update(prec1_final[0], input_a.size(0))

        

        clip_name_lst.extend(path)
        target_lst.extend(target.tolist())
        predict_lst.extend(pred.tolist())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('---Test: [{0}/{1}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'acc_f {top1_f.avg:.3f} acc_a {top1_a.avg:.3f} {top1_final.avg:.3f}'.format(
                  # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  # 'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1_f=top1_f, top1_a=top1_a, top1_final=top1_final))

    print(' * Val Acc@1 {top1.avg:.3f} '.format(top1=top1_final))

    clip_name_lst = [pathlib.Path(clipname).stem for clipname in clip_name_lst]
    df = pd.DataFrame({'clip':clip_name_lst, 'target':target_lst, 'pred':predict_lst}) 


 
    # print(clip_name_lst)
    # print(target_lst)
    # print(predict_lst)

    return top1_final.avg.cpu().item(), df


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred[0]

def accuracy_fusion(output_a, output_f, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    # tar = target.unsqueeze(1)
    # print('----'*10, 'output_a')
    # out_a= torch.cat([output_a, tar],dim=1)
    # print(out_a)

    # print('----'*10, 'output_f')
    # out_f= torch.cat([output_f, tar],dim=1)
    # print(out_f)

    output = 0.4*output_a + 0.6*output_f
    # print('----'*10, 'output')
    # out= torch.cat([output, tar],dim=1)
    # print(out)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred[0]

def accuracy_fusion1(output_a, output_f, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred_a = output_a.topk(maxk, 1, True, True)
    pred_a = pred_a.t()
    correct_a = pred_a.eq(target.view(1, -1).expand_as(pred_a))

    _, pred_f = output_f.topk(maxk, 1, True, True)
    pred_f = pred_f.t()
    correct_f = pred_f.eq(target.view(1, -1).expand_as(pred_f))

    correct = correct_a & correct_f

    # print(correct_a, correct_f, correct)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred_f[0]

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

if __name__ == '__main__':
    main()
