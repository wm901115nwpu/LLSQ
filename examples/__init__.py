import argparse
import json
import os
import shutil
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

import models._modules as my_nn
from utils import wrapper
from utils.dnq import dnq_scheduler
from utils.ptflops import get_model_complexity_info

str_q_mode_map = {'layer_wise': my_nn.Qmodes.layer_wise,
                  'kernel_wise': my_nn.Qmodes.kernel_wise}


def get_base_parser():
    """
        Default values should keep stable.
    """

    print('Please do not import ipdb when using distributed training')

    q_modes_choice = sorted(['kernel_wise', 'layer_wise'])
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # ==========learning rate schedule<<<<<<<<<<<<<
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--warmup-epoch', type=int, default=-3, help='warm up epoch')
    parser.add_argument('--warmup-multiplier', type=float, default=10, help='warm up multiplier')

    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')

    lr_scheduler_choice = ['StepLR', 'MultiStepLR', 'CosineAnnealingLR']
    parser.add_argument('--lr-scheduler', default='CosineAnnealingLR', choices=lr_scheduler_choice)

    parser.add_argument('--step-size', default=20, type=int,
                        help='step size of StepLR')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='lr decay of StepLR or MultiStepLR')
    parser.add_argument('--milestones', default=[30, 100, 200], nargs='+', type=int,
                        help='milestones of MultiStepLR')
    # >>>>>>>>>>>>>>>learning rate schedule END=============

    # =========================DNQ<<<<<<<<<<<<<<<<<<<<<<<<<<<
    dnq_scheduler_choice = ['MultiStepDNQ']
    parser.add_argument('--dnq-scheduler', default='MultiStepDNQ', choices=dnq_scheduler_choice)
    parser.add_argument('--dnq-gamma', default=2, type=float)
    parser.add_argument('--dnq-milestones', default=[100], nargs='+', type=int,
                        help='milestones of MultiStepDNQ')
    # >>>>>>>>>>>>>>>>>>>>>>>>DNQ END=========================

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume-after', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--log-name', default='log', type=str)

    # =====================Generate model key map<<<<<<<<<<<<<<<<<<<<<<<<<<<
    parser.add_argument('--gen-map', action='store_true', default=False,
                        help='generate key map for quantized model')
    parser.add_argument('--original-model', default='', type=str,
                        help='original model')
    # >>>>>>>>>>>>>>>>>>>Generate model key map END=======================

    parser.add_argument('--bn-fusion', action='store_true', default=False,
                        help='ConvQ + BN fusion')
    parser.add_argument('--resave', action='store_true', default=False,
                        help='resave the model')

    parser.add_argument('--quant-bias-scale', action='store_true', default=False,
                        help='Add Qcode for scale and quantize bias')
    parser.add_argument('--extract-inner-data', action='store_true', default=False,
                        help='Extract inner feature map and weights')
    parser.add_argument('--export-onnx', action='store_true', default=False,
                        help='Export model to onnx')

    # ===========================multi-gpu training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--gpus', default=None, type=str,
                        help='GPUs id to use.You can specify multiple GPUs separated by ,')
    # >>>>>>>>>>>>>>>>>>>>>>>>>multi-gpu training END==============================

    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--qa', default=0, type=int,
                        help='quantized activation bit')
    parser.add_argument('--qw', default=0, type=int,
                        help='quantized weight bit')
    parser.add_argument('--q-mode', choices=q_modes_choice, default='kernel_wise',
                        help='Quantization modes: ' +
                             ' | '.join(q_modes_choice) +
                             ' (default: kernel-wise)')
    parser.add_argument('--l1', action='store_true', default=False,
                        help='Use l1 error to optimize parameter of quantizer (default: l2)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='save running scale in tensorboard')
    parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN')
    return parser


def get_lr_scheduler(optimizer, args):
    # todo: add different scheduler
    if args.lr_scheduler == 'CosineAnnealingLR':
        print('Use cosine scheduler')
        scheduler_next = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'StepLR':
        print('Use step scheduler, step size: {}, gamma: {}'.format(args.step_size, args.gamma))
        scheduler_next = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        print('Use MultiStepLR scheduler, milestones: {}, gamma: {}'.format(args.milestones, args.gamma))
        scheduler_next = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        raise NotImplementedError
    if args.warmup_epoch <= 0:
        return scheduler_next
    print('Use warmup scheduler')
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=args.warmup_multiplier,
                                          total_epoch=args.warmup_epoch,
                                          after_scheduler=scheduler_next)
    return lr_scheduler


def get_dnq_scheduler(model, args):
    print('Use MultiStepDNQ scheduler, milestones: {}, gamma: {}'.format(args.dnq_milestones, args.dnq_gamma))
    scheduler = dnq_scheduler.MultiStepDNQ(model, milestones=args.dnq_milestones, gamma=args.dnq_gamma, nbits=args.qw)
    return scheduler


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def gen_key_map(new_dict, old_dict):
    new_keys = [(k, v.size()) for (k, v) in new_dict.items()]
    ori_keys = [(k, v.size()) for (k, v) in old_dict.items()]
    key_map = OrderedDict()
    assert len(new_keys) == len(ori_keys)
    for i in range(len(new_keys)):
        if 'expand_' in new_keys[i][0] and ori_keys[i][1] != new_keys[i][1]:
            print('{}({}) is expanded to {}({})'.format(ori_keys[i][0], ori_keys[i][1],
                                                        new_keys[i][0], new_keys[i][1]))
        else:
            assert ori_keys[i][1] == new_keys[i][1], '{} != {}'.format(ori_keys[i][1], new_keys[i][1])
        key_map[new_keys[i][0]] = ori_keys[i][0]
        print('{} <==> {}'.format(new_keys[i][0], ori_keys[i][0]))
    return key_map


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
    return


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if args.extract_inner_data:
                print('early stop evaluation')
                break
            if i % args.print_freq == 0:
                progress.print(i)

        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5))

    return top1.avg, top5.avg


def is_finish(finishes):
    finish = True
    for f in finishes:
        finish = finish and f
    return finish


def train(train_loader, model, criterion, optimizer, epoch, args, writer, criterion_admm=None,
          admm_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    if args.freeze_bn:
        model.apply(set_bn_eval)
    end = time.time()
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        elif args.gpus is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        # compute output
        output = model(inputs)
        loss = criterion(output, targets)
        if criterion_admm is not None:
            loss += criterion_admm(model, admm_scheduler.Z, admm_scheduler.U)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
        # compute gradient and do SGD step
        # optimizer.param_groups[0]['params']:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # warning 1. backward 2. step 3. zero_grad
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
            if writer is not None and args.debug:
                for k, v in model.state_dict().items():
                    if 'module.' in k and args.gpus is not None:
                        k = k[7:]
                    if 'alpha' in k or 'scale' in k:
                        if v.shape[0] == 1:
                            writer.add_scalar('train/{}/{}'.format(args.arch, k), v.item(), base_step + i)
                        else:
                            writer.add_histogram('train/{}/{}'.format(args.arch, k), v, base_step + i)
    return


def get_summary_writer(args, ngpus_per_node):
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        if 'q' in args.arch:
            args.log_name = 'logger/{}_w{}a{}_{}'.format(args.arch, args.qw, args.qa,
                                                         args.log_name)
        else:
            args.log_name = 'logger/{}_{}'.format(args.arch,
                                                  args.log_name)
        writer = SummaryWriter(args.log_name)
        return writer
    return None


def main_gen_key_map(args, model, models):
    if isinstance(models, list):
        assert len(models) == 2, len(models)
        assert args.original_model in models[0].__dict__ or args.original_model in models[
            1].__dict__, '{} must be included'.format(args.original_model)
        try:
            ori_model = models[0].__dict__[args.original_model]()
        except KeyError:
            ori_model = models[1].__dict__[args.original_model]()
    else:
        assert args.original_model in models.__dict__, '{} must be included'.format(args.original_model)
        ori_model = models.__dict__[args.original_model]()
    print('Original model:\n=========\n{}\n=========='.format(ori_model))
    key_map = gen_key_map(model.state_dict(), ori_model.state_dict())
    with open('models/weight_keys_map/{}_map.json'.format(args.arch), 'w') as wf:
        json.dump(key_map, wf)
    print('Generate key map done')
    return


def get_model_info(model, args, input_size=(3, 224, 224)):
    print('Inference for complexity summary')
    if isinstance(input_size, torch.utils.data.DataLoader):
        input_size = input_size.dataset.__getitem__(0)[0].shape
        input_size = (input_size[0], input_size[1], input_size[2])
    with open('{}/{}_flops.txt'.format(args.log_name, args.arch), 'w') as f:
        try:
            flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True,
                                                      ost=f)
        except:
            print('get model info error')
            return None
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    with open('{}/{}.txt'.format(args.log_name, args.arch), 'w') as wf:
        wf.write(str(model))
    # summary(model, input_size)
    if args.export_onnx:
        dummy_input = torch.randn(1, input_size[0], input_size[1], input_size[2], requires_grad=True).cuda(args.gpu)
        # torch_out = model(dummy_input)
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          "{}.onnx".format(args.arch),  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          # opset_version=10,  # the ONNX version to export the model to
                          input_names=['input'],  # the model's input names
                          output_names=['output']  # the model's output names
                          )
    return flops, params


def save_checkpoint(state, is_best, prefix, filename='checkpoint.pth.tar'):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'best.pth.tar')
    return


def process_model(model, optimizer, args, conv_name=None, **kwargs_conv):
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])  # GPU memory leak. todo

            if not args.quant_bias_scale:
                # todo: del
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}) (acc: {})"
                      .format(args.resume, checkpoint['epoch'], best_acc1))

                if args.resave:
                    model.cpu()
                    print('=> save only weights in {}.pth'.format(args.arch))
                    torch.save(model.state_dict(), '{}.pth'.format(args.arch))
                model.cuda(args.gpu)
                # save pth here
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if conv_name is not None:
        print('replace conv {}'.format(conv_name))
        wrapper.replace_conv_recursively(model, conv_name, **kwargs_conv)
        print('after conv replacement')
        print(model)
        args.arch = '{}_{}'.format(args.arch, conv_name)

    if args.bn_fusion:
        print('BN fusion begin')
        model = wrapper.fuse_bn_recursively(model)
        print('after bn fusion: ')
        print(model)
        if args.resave:
            print('=> re-save the weights in {}_wo_bn.pth'.format(args.arch))
            torch.save(model.state_dict(), '{}_wo_bn.pth'.format(args.arch))

    if args.quant_bias_scale:
        # todo: del this
        print('add qcode for scale and quantize bias')
        model = wrapper.quantize_scale_and_bias(model)
        print('after quantize scale and bias')
        print(model)

    if args.resume_after:
        if os.path.isfile(args.resume_after):
            print('=> loading checkpoint {}'.format(args.resume_after))
            checkpoint = torch.load(args.resume_after, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.cuda(args.gpu)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.extract_inner_data:
        print('extract inner feature map and weight')
        wrapper.save_inner_hooks(model)
        if not args.evaluate:
            warnings.warn('When extract_inner_data is true, -e is recommended')
            args.evaluate = True
        for k, v in model.state_dict().items():
            print('saving {}'.format(k))
            np.save('{}'.format(k), v.cpu().numpy())
    return


class DataloaderFactory(object):
    cifar10 = 1
    cifar10_positive_shift = 2
    imagenet2012 = 5
    caltech101 = 7
    cifar100 = 8
    flower102 = 9

    def __init__(self, args):
        self.args = args
        self.cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.cifar10_transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.cifar10_positive_shift_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (0.25, 0.25, 0.25)),
        ])
        self.cifar10_positive_shift_transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (0.25, 0.25, 0.25)),
        ])

    def product_train_val_loader(self, data_type):
        args = self.args
        train_loader = None
        val_loader = None
        if data_type == self.cifar10:
            trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,
                                                    transform=self.cifar10_transform_train)
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            else:
                train_sampler = None
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                       shuffle=(train_sampler is None),
                                                       num_workers=args.workers, sampler=train_sampler)
            testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                                   transform=self.cifar10_transform_val)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
        elif data_type == self.cifar10_positive_shift:
            trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,
                                                    transform=self.cifar10_positive_shift_transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                                   transform=self.cifar10_positive_shift_transform_val)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
        elif data_type == self.imagenet2012:
            # Data loading code
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
            args.batch_num = len(train_loader)

            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=64, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            # todo: args.batch_size
            return train_loader, val_loader, train_sampler

        elif data_type == self.caltech101:
            transform_train = transforms.Compose([
                transforms.Resize(36),
                transforms.RandomCrop(32),
                # transforms.Resize(72),
                # transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize([32, 32]),
                # transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_data = torchvision.datasets.ImageFolder(args.data + '/train', transform=transform_train)
            valid_data = torchvision.datasets.ImageFolder(args.data + '/val', transform=transform_test)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.workers)
        elif data_type == self.cifar100:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                                     transform=transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            args.batch_num = len(train_loader)
            testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                                    transform=transform_test)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
        elif data_type == self.flower102:
            # The input size can be set to 224
            # https://github.com/MiguelAMartinez/flowers-image-classifier/blob/master/utils_ic.py#L19
            # transform_train = transforms.Compose([transforms.RandomRotation(30),
            #                                       transforms.Resize(255),
            #                                       transforms.CenterCrop(224),
            #                                       transforms.RandomHorizontalFlip(),
            #                                       transforms.ToTensor(),
            #                                       transforms.Normalize([0.485, 0.456, 0.406],
            #                                                            [0.229, 0.224, 0.225])])
            #
            # transform_test = transforms.Compose([transforms.Resize(255),
            #                                      transforms.CenterCrop(224),
            #                                      transforms.ToTensor(),
            #                                      transforms.Normalize([0.485, 0.456, 0.406],
            #                                                           [0.229, 0.224, 0.225])])

            transform_train = transforms.Compose([
                transforms.Resize(36),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize([32, 32]),
                # transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_data = torchvision.datasets.ImageFolder(args.data + '/train', transform=transform_train)
            test_data = torchvision.datasets.ImageFolder(args.data + '/valid', transform=transform_test)
            valid_data = torchvision.datasets.ImageFolder(args.data + '/test', transform=transform_test)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.workers)
        else:
            assert NotImplementedError
        return train_loader, val_loader


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def distributed_model(model, ngpus_per_node, args):
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model
