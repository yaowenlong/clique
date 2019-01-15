import argparse
import shutil
import datetime
import time
import random
import os


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data import MydataSet
from vgg import vgg16
import imagenet_pytorch.cliquenet as cliquenet


parser = argparse.ArgumentParser(description='CliqueNet ImageNet Training')



parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 160)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--no-attention', dest='attention', action='store_false', help='not use attentional transition')
parser.set_defaults(attention=True)

best_prec1 = 0

path = "/data/datasets/"
txtPath = "/data/datasets/fish"

pwd = os.getcwd()

# pa = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
pa = pwd
imgPath = pa+path
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
datasetTrain = MydataSet(pa + txtPath + "/train.txt",imgPath, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
datasetTest = MydataSet(pa + txtPath + "/test_list.txt",imgPath, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
def main():

    global args, best_prec1
    args = parser.parse_args()
    if args.attention:
        print ('attentional transition is used')
    # create model
    model = cliquenet.build_cliquenet(input_channels=64, list_channels=[40, 80, 160, 160], list_layer_num=[6, 6, 6, 6], if_att=args.attention,w=torch.tensor([1.0]).cuda())
    print("teacher model")
    print(model)
    modelstu = cliquenet.build_cliquenet(input_channels=64, list_channels=[40, 80, 160, 160], list_layer_num=[3, 3, 3, 3], if_att=False,w=torch.tensor([1.0]).cuda())
    print("stu model")
    print(modelstu)
    # model = cliquenet()
    # model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizerstu = torch.optim.SGD(modelstu.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

   # # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
       datasetTrain,
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasetTest,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model,modelstu, criterion)
        return

    # get_number_of_param(model)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, modelstu,criterion, optimizer,optimizerstu, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model,modelstu, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

def get_number_of_param(model):
    """get the number of param for every element"""
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for dis in param_size:
            count_of_one_param *= dis
        print(param.size(), count_of_one_param)
        print(count)
        count += count_of_one_param
    print('total number of the model is %d'%count)


def train(train_loader, model,modelstu, criterion, optimizer,optimizerstu, epoch):
    """train model"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    losses = AverageMeter()
    lossesstu = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1stu = AverageMeter()
    top5stu = AverageMeter()

    # switch to train mode
    model.train()
    modelstu.train()
    end = time.time()

    # last_datetime = datetime.datetime.now()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        
        data_time.update(time.time() - end)
        # input_var = input.cuda()
        input_var = input.cuda()
        # pdb.set_trace()

        target_var = target.cuda()
        model = model.cuda()
        
        # compute output
        print("teacher")
        output,w = model(input_var)
        modelstu.w = w
        modelstu = modelstu.cuda()
        print("stu")
        outputstu,wstu = modelstu(input_var)
        a = target_var[0][0]
        
        a = torch.tensor([a]).cuda()
        loss = criterion(output,a)
        lossstu = criterion(outputstu,a)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, a, topk=(1, 5))
        prec1stu, prec5stu = accuracy(outputstu.data, a, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        losses.update(lossstu.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        top1stu.update(prec1stu[0], input.size(0))
        top5stu.update(prec5stu[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        optimizerstu.zero_grad()
        loss.backward(retain_variables=True)
        lossstu.backward(retain_variables=True)
        optimizer.step()
        optimizerstu.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  'Lossstu {lossstu.val:.4f} ({lossstu.avg:.4f})\t'
                  'Prec@1stu {top1stu.val:.3f} ({top1stu.avg:.3f})\t'
                  'Prec@5stu {top5stu.val:.3f} ({top5stu.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5,lossesstu=lossesstu,top1stu=top1stu,top5stu=top5stu))

    print (time.ctime())


def validate(val_loader, model,modelstu, criterion):
    """validate model"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    lossesstu = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1stu = AverageMeter()
    top5stu = AverageMeter()

    # switch to evaluate mode
    model.eval()
    modelstu.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()
        model = model.cuda()
        modelstu.if_att = True
        modelstu = modelstu.cuda()

        # compute output
        print("teacher")
        output,w = model(input_var)
        print("stu")
        outputstu,wstu = modelstu(input_var)
        a = torch.tensor([a]).cuda()
        loss = criterion(output,a)
        lossstu = criterion(outputstu,a)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, a, topk=(1, 5))
        prec1stu, prec5stu = accuracy(outputstu.data, a, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        lossesstu.update(lossstu.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        top1stu.update(prec1stu[0], input.size(0))
        top5stu.update(prec5stu[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  'Lossstu {lossstu.val:.4f} ({lossstu.avg:.4f})\t'
                  'Prec@1 {top1stu.val:.3f} ({top1stu.avg:.3f})\t'
                  'Prec@5 {top5stu.val:.3f} ({top5stu.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5,lossesstu=lossesstu,top1stu = top1stu,top5stu=top5stu))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1stu, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='./cliquenet_s0.pth.tar'):
    """Save the trained model"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './best_cliquenet_s0.pth.tar')


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
    """Sets the learning rate to the initial Learning rate decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('current learning rate is: %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    return res


if __name__ == '__main__':
    main()
