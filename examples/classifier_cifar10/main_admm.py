'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import random

import torch.backends.cudnn as cudnn
import torch.nn as nn

import models.cifar10 as cifar10_models
from examples import *
from utils.admm import AdmmPercentagePrunerScheduler

model_names = sorted(name for name in cifar10_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(cifar10_models.__dict__[name]))

best_acc1 = 0


def main():
    parser = get_base_parser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg10_cifar10',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: vgg10_cifar10)')
    parser.add_argument('--sparsity', default=0, type=float, help='the expected sparsity of each layer (default: 0.0)')
    parser.add_argument('--non-zero-num', default=32, type=int,
                        help='the non-zero-num of each pe array in NPU (default: 32)')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R',
                        help='cardinality weight (default: 1e-2)')
    parser.add_argument('--rho-dynamic', action='store_true', default=False,
                        help='whether to adjust rho dynamically in a heuristic way')
    parser.add_argument('--admm', action='store_true', default=False,
                        help='whether to adopt admm step1')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.gpus is None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # Simply call main_worker function
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    if args.gen_map:
        args.qw = -1
        args.qa = -1
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    try:
        model = cifar10_models.__dict__[args.arch](pretrained=args.pretrained,
                                                   )
    except KeyError:
        print('do not support {}'.format(args.arch))
        return

    print('model:\n=========\n{}\n=========='.format(model))
    if args.gen_map:
        main_gen_key_map(args, model, cifar10_models)
        return

    if args.gpu is not None and args.gpus is None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print('Use {} gpus'.format(args.gpus))
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    params = add_weight_decay(model, weight_decay=args.weight_decay, skip_keys=['alpha'])
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    process_model(model, optimizer, args)

    cudnn.benchmark = True

    # Data loading code
    print('==> Preparing data..')
    df = DataloaderFactory(args)
    train_loader, val_loader = df.product_train_val_loader(df.cifar10)
    writer = get_summary_writer(args)
    if (args.qw <= 0 and args.qa <= 0) or args.evaluate:
        get_model_info(model, args, val_loader)
    args.batch_num = len(train_loader)

    scheduler_lr = get_lr_scheduler(optimizer, args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.admm:
        print('ADMM Begin')
        criterion_admm = my_nn.AdmmLoss(args.rho).cuda(args.gpu)  # todo:test
        scheduler_admm = AdmmPercentagePrunerScheduler(model, args.sparsity)  # init, U, Z
        acc_bl, _ = validate(val_loader, model, criterion, args)
        for epoch in range(args.start_epoch, args.epochs):
            # adjust_learning_rate(optimizer, epoch, args)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, writer, criterion_admm, scheduler_admm)
            convergence = scheduler_admm.update_per_epoch()
            scheduler_lr.step()

            # evaluate on validation set
            acc1, _ = validate(val_loader, model, criterion, args)
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('val/convergence', convergence, epoch)
            writer.add_scalar('val/rho', criterion_admm.rho, epoch)
            if args.rho_dynamic:
                criterion_admm.adjust_rho(convergence, acc1 / acc_bl, epoch / args.epochs)
            if args.debug:
                idx = 0
                for name, param in model.named_parameters():
                    if name.split('.')[-1] == "weight" and len(param.shape) != 1:
                        writer.add_histogram(name, param, epoch)
                        writer.add_histogram('{}_Z'.format(name), scheduler_admm.Z[idx], epoch)
                        idx += 1
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1 and convergence < 0.01
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, prefix='{}/{}_'.format(args.log_name, args.arch))
    else:
        # retrain
        wrapper.replace_conv_recursively(model, 'Conv2dSQ', nbits_a=-1, nbits_w=-1, sparsity=args.sparsity,
                                         total_iter=args.batch_num * args.epochs, INS=False, beta=0)
        print(model)
        validate(val_loader, model, criterion, args)
        for epoch in range(args.start_epoch, args.epochs):
            # adjust_learning_rate(optimizer, epoch, args)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, writer)
            scheduler_lr.step()

            # evaluate on validation set
            acc1, _ = validate(val_loader, model, criterion, args)
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
            if args.debug:
                for module_name, module in model.named_modules():
                    if isinstance(module, my_nn.Conv2dDNQ):
                        writer.add_scalar('val/nbits', module.nbits, epoch)
                        break
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, prefix='{}/{}_'.format(args.log_name, args.arch))


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    main()
