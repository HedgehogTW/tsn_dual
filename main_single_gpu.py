import os
import pathlib
import time, datetime
import argparse
import shutil
import pickle
import numpy as np
import pandas as pd

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

# for evaluation: $ python main_single_gpu.py E:/tsn_data -e
# for training: $ python main_single_gpu.py d:/tsn_data -m rgb
parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('data', metavar='DIR', 
                    help='path to dataset')
# parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
#                     help='path to datset setting files')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=["rgb", "flow"],
                    help='modality: rgb | flow')
parser.add_argument('--dataset', '-d', default='rat',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_vgg16)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter_size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
# parser.add_argument('--new_length', default=1, type=int,
#                     metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
#                     metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[80, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=20, type=int,
                    metavar='N', help='save frequency (default: 25)')
# parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--opti', '-o', default='adam',
                    choices=["sgd", "adam"],
                    help='optimizer: sgd | adam')

best_prec1 = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.arch[:3]=='rgb':
        args.modality = "rgb"
        args.new_length = 1
    else:
        args.modality = "flow"
        args.new_length = 10

    print('support model:', model_names)
    # create model
    print("Building model ...arch {}, {}, {} ".format(args.arch, args.modality, args.opti ))
    print('new_length for {} is {}'.format(args.modality, args.new_length))
    print('--evaluate ', args.evaluate, ', modality: ', args.modality)

    model = build_model()
    print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.opti =='sgd':
        args.lr = 0.001
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opti =='adam':
        args.lr = 0.0005 #0.001 #2e-4
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(0.9,0.99),
                        eps=1e-08, weight_decay=0)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    print(lr_scheduler.__dict__)
    
    args.resume = os.path.join(args.data, 'checkpoints')
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("checkpoints directory %s." % (args.resume))

    cudnn.benchmark = True

    # Data transforming
    if args.modality == "rgb":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.new_length
        clip_std = [0.229, 0.224, 0.225] * args.new_length
    elif args.modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * args.new_length
        clip_std = [0.226, 0.226] * args.new_length
    else:
        print("No such modality. Only rgb and flow supported.")

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.MultiScaleCrop((224, 224), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.CenterCrop((224)),
            video_transforms.ToTensor(),
            normalize,
        ])

    # data loading
    train_setting_file = "train_%s_split%d.txt" % (args.modality, args.split)
    train_split_file = os.path.join(args.data, args.dataset, train_setting_file)

    val_setting_file = "val_%s_split%d.txt" % (args.modality, args.split)
    val_split_file = os.path.join(args.data, args.dataset, val_setting_file)

    test_setting_file = "test_%s_split%d.txt" % (args.modality, args.split)
    test_split_file = os.path.join(args.data, args.dataset, test_setting_file)

    if not os.path.exists(train_split_file) or not os.path.exists(test_setting_file) \
    or not os.path.exists(test_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.data))


    if not args.evaluate:
        train_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                        source=train_split_file,
                                                        phase="train",
                                                        modality=args.modality,
                                                        is_color=is_color,
                                                        new_length=args.new_length,
                                                        new_width=args.new_width,
                                                        new_height=args.new_height,
                                                        video_transform=train_transform)
        val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                      source=val_split_file,
                                                      phase="val",
                                                      modality=args.modality,
                                                      is_color=is_color,
                                                      new_length=args.new_length,
                                                      new_width=args.new_width,
                                                      new_height=args.new_height,
                                                      video_transform=val_transform)

        print('{} samples found, {} train samples and {} val samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

    else:
        test_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                      source=test_split_file,
                                                      phase="val",
                                                      modality=args.modality,
                                                      is_color=is_color,
                                                      new_length=args.new_length,
                                                      new_width=args.new_width,
                                                      new_height=args.new_height,
                                                      video_transform=val_transform)

        print('{} test samples found in {}'.format(len(test_dataset), test_split_file))

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    ################################ evaluate
    if args.evaluate:
        path_checkpoint = pathlib.Path(args.resume )
        
        chk_lst = sorted(path_checkpoint.glob('checkpoint_{}*.tar'.format(args.modality)))
        checkpoint_file = chk_lst[-1]
        print('checkpoint_file:', checkpoint_file)
        best_model_name = os.path.join(args.resume, checkpoint_file)
        if os.path.isfile(best_model_name):
            print("==> loading checkpoint '{}'".format(best_model_name))
            checkpoint = torch.load(best_model_name)
            args.arch = checkpoint['arch']
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
              .format(args.arch, args.start_epoch, best_prec1))
        else:
            print("==> no checkpoint found at '{}'".format(best_model_name)) 

        prec1, df = validate(test_loader, model, criterion)
        dt = datetime.datetime.now().strftime("%m%d_%H%M")
        out_name = 'test_{}_{}_{}.csv'.format(args.arch, args.opti, dt)
        df.to_csv(out_name,  sep = ',')
        print('acc: {:.4f}, out file: {}'.format(prec1, out_name))
        return

    # return

    train_acc = []
    valid_acc = []
    learn_rate_lst = []

    t0 = time.time()

    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # lr = adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        learn_rate_lst.append(lr)
        print("Current learning rate is {:4.6f}, opti {}".format(lr, args.opti ))

        # train for one epoch
        train_prec1 = train(train_loader, model, criterion, optimizer, epoch)
        train_acc.append(train_prec1.cpu().item()/100)

        lr_scheduler.step()

        # evaluate on validation set
        valid_freq = 1 #args.save_freq //2
        prec1 = 0.0
        if (epoch + 1) % valid_freq == 0:
            prec1, _ = validate(val_loader, model, criterion)
            valid_acc.append(prec1/100)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # if (epoch + 1) % args.save_freq == 0:
        #     checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, checkpoint_name, args.resume)

        epoch_time = format_time(time.time() - end)
        end = time.time()
        print("  Epcoh {} took: {:}".format(epoch + 1, epoch_time))




    training_time = format_time(time.time() - t0)
    print("Training epcoh took: {:}".format(training_time))

    # df_hist= pd.DataFrame({'train_acc':train_acc, 'valid_acc':valid_acc})

    opt = lr_scheduler.__dict__.pop('optimizer')
    lr_scheduler.__dict__['optimizer'] = str(opt)

    hist = {'train_acc': train_acc, 'valid_acc': valid_acc, 'lr':learn_rate_lst, 'opti':args.opti, 
            'arch': args.arch, 'training_time':training_time, 'lr_scheduler': lr_scheduler.__dict__}

    dt = datetime.datetime.now().strftime("%m%d_%H%M")
    out_name = 'hist_{}_{}_{}.pkl'.format(args.arch, args.opti, dt)
    # df_hist.to_csv(out_name)
    f = open(out_name, 'wb')
    pickle.dump(hist, f) # dump the object to a file 
    f.close()
    print('Training history:', out_name)

    checkpoint_name = "checkpoint_{}_{}_{}.pth.tar".format(args.arch, args.opti, dt)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, False, checkpoint_name, args.resume)


def build_model():

    model = models.__dict__[args.arch](pretrained=True, num_classes=2)
    model.cuda()
    return model

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    acc_mini_batch = 0.0

    for i, (input, target, index, path) in enumerate(train_loader):
        input = input.float().cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)

        # measure accuracy and record loss
        prec1, _ = accuracy(output.data, target) #, topk=(1, 3))
        acc_mini_batch += prec1[0]
        loss = criterion(output, target_var)
        loss = loss / args.iter_size
        loss_mini_batch += loss.data
        loss.backward()

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

            # losses.update(loss_mini_batch/args.iter_size, input.size(0))
            # top1.update(acc_mini_batch/args.iter_size, input.size(0))
            losses.update(loss_mini_batch, input.size(0))
            top1.update(acc_mini_batch/args.iter_size, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch = 0
            acc_mini_batch = 0

            if (i+1) % args.print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))

    return top1.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    clip_name_lst = []
    target_lst = []
    predict_lst = []
    end = time.time()
    for i, (input, target, index, path) in enumerate(val_loader):
        # print(index, path, target)

        input_var = input.float().cuda(non_blocking=True)
        target_var = target.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        with torch.no_grad():
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)



        # measure accuracy and record loss
        prec1, pred = accuracy(output.data, target_var) #, topk=(1, 3))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        # top3.update(prec3, input.size(0))

        clip_name_lst.extend(path)
        target_lst.extend(target.tolist())
        predict_lst.extend(pred.tolist())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('---Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  # 'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Val Prec@1 {top1.avg:.3f} '.format(top1=top1))

    clip_name_lst = [pathlib.Path(clipname).stem for clipname in clip_name_lst]
    df = pd.DataFrame({'clip':clip_name_lst, 'target':target_lst, 'pred':predict_lst}) 

 
    # print(clip_name_lst)
    # print(target_lst)
    # print(predict_lst)

    return top1.avg.cpu().item(), df

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

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
