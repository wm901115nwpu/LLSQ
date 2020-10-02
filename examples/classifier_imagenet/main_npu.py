from examples import *
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import torch.multiprocessing as mp
import torch.distributed as dist
import models.imagenet as imagenet_extra_models
from utils.admm import AdmmNpuScheduler

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.extend(sorted(name for name in imagenet_extra_models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(imagenet_extra_models.__dict__[name])))
best_acc1 = 0


def main():
    parser = get_base_parser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--non-zero-num', default=32, type=int,
                        help='the non-zero-num of each pe array in NPU (default: 32)')
    parser.add_argument('--INS', action='store_true', default=False,
                        help='incremental network sparse')
    parser.add_argument('--beta', default=0.1, type=float,
                        help='The proportion of the INS phase in the training process (default: 0.1)')

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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.gen_map:
        args.qw = -1
        args.qa = -1
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    try:
        model = models.__dict__[args.arch](pretrained=args.pretrained)
    except KeyError:
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', args.arch, pretrained=args.pretrained)
    print('model:\n=========\n{}\n=========='.format(model))

    # parallel and multi-gpu
    model = distributed_model(model, ngpus_per_node, args)

    writer = get_summary_writer(args, ngpus_per_node)

    df = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = df.product_train_val_loader(df.imagenet2012)
    args.batch_num = len(train_loader)
    if writer is not None:
        get_model_info(model, args, val_loader)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    params = add_weight_decay(model, weight_decay=args.weight_decay, skip_keys=['expand_', 'scale', 'alpha'])
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum)
    if args.admm:
        process_model(model, optimizer, args)
    else:
        process_model(model, optimizer, args, 'Conv2dNPU', non_zero_num=args.non_zero_num,
                      total_iter=args.batch_num * args.epochs, INS=args.INS, beta=args.beta)

    cudnn.benchmark = True

    scheduler = get_lr_scheduler(optimizer, args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        get_model_info(model, args, val_loader)
        return

    for epoch in range(0, args.start_epoch):
        scheduler.step()
        pass
    if args.admm:
        print('ADMM Begin')
        criterion_admm = my_nn.AdmmLoss(args.rho).cuda(args.gpu)
        scheduler_admm = AdmmNpuScheduler(model, args.non_zero_num)  # init, U, Z
        if args.arch == 'resnet18':
            acc_bl = 69.758
        else:
            acc_bl, _ = validate(val_loader, model, criterion, args)
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # adjust_learning_rate(optimizer, epoch, args)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, writer, criterion_admm, scheduler_admm)
            convergence = scheduler_admm.update_per_epoch()
            scheduler.step()
            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args)
            if writer is not None:
                writer.add_scalar('val/acc1', acc1, epoch)
                writer.add_scalar('val/acc5', acc5, epoch)
                writer.add_scalar('val/convergence', convergence, epoch)
                writer.add_scalar('val/rho', criterion_admm.rho, epoch)
            if args.rho_dynamic:
                criterion_admm.adjust_rho(convergence, acc1 / acc_bl, epoch / args.epochs)

            if args.debug and writer is not None:
                idx = 0
                for name, param in model.named_parameters():
                    if name.split('.')[-1] == "weight" and len(param.shape) != 1:  # len(BN.weight) = 1
                        if 'module.' in name:
                            name = name[7:]
                        writer.add_histogram(name, param, epoch)
                        writer.add_histogram('{}_Z'.format(name), scheduler_admm.Z[idx], epoch)
                        idx += 1
            # remember best acc@1 and save checkpoint
            is_best = (acc1 > best_acc1) and convergence < 0.01
            best_acc1 = max(acc1, best_acc1)
            #
            if writer is not None:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, prefix='{}/{}'.format(args.log_name, args.arch))
    else:
        # retrain
        if not args.INS:
            validate(val_loader, model, criterion, args)
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # adjust_learning_rate(optimizer, epoch, args)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, writer)
            scheduler.step()
            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args)
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/acc5', acc5, epoch)
            if args.debug:
                get_model_info(model, args, val_loader)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            #
            if not args.multiprocessing_distributed:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, prefix='{}/{}'.format(args.log_name, args.arch))
    writer.close()


if __name__ == '__main__':
    main()
