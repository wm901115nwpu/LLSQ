import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import models.imagenet as imagenet_extra_models
from examples import *
from utils.admm import AdmmPercentagePrunerScheduler

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.extend(sorted(name for name in imagenet_extra_models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(imagenet_extra_models.__dict__[name])))
model_names.extend(['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                    'efficientnet_b3', 'efficientnet_b3', 'efficientnet_b4',
                    'efficientnet_b5', 'efficientnet_b6'])
best_acc1 = 0


def main():
    parser = get_base_parser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--sparsity', default=0.0, type=float,
                        help='sparsity level (default: 0.0)')
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

    # Simply call main_worker function
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print('Use {} gpus'.format(args.gpus))
        ngpus = torch.cuda.device_count()  #
        print('ngpus: {}'.format(ngpus))
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    writer = get_summary_writer(args)

    df = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = df.product_train_val_loader(df.imagenet2012)
    args.batch_num = len(train_loader)

    get_model_info(model, args, val_loader)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    params = add_weight_decay(model, weight_decay=args.weight_decay, skip_keys=['expand_', 'scale', 'alpha'])
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum)

    process_model(model, optimizer, args)

    cudnn.benchmark = True

    scheduler = get_lr_scheduler(optimizer, args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        get_model_info(model, args, val_loader)
        print('s: {} w{}a{}'.format(args.sparsity, args.qw, args.qa))
        return

    for epoch in range(0, args.start_epoch):
        scheduler.step()
        pass
    if args.admm:
        print('ADMM Begin')
        criterion_admm = my_nn.AdmmLoss(args.rho).cuda(args.gpu)
        scheduler_admm = AdmmPercentagePrunerScheduler(model, args.sparsity, prune_linear=False,
                                                       prune_first_layer=False)  # init, U, Z
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
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/acc5', acc5, epoch)
            writer.add_scalar('val/convergence', convergence, epoch)
            writer.add_scalar('val/rho', criterion_admm.rho, epoch)
            if args.rho_dynamic:
                criterion_admm.adjust_rho(convergence, acc1 / acc_bl, epoch / args.epochs)

            if args.debug:
                for module_name, module in model.named_modules():
                    if isinstance(module, my_nn.Conv2dSQ):
                        writer.add_scalar('val/sparsity', module.get_sparsity(module.iter), epoch)
                        break
                    if isinstance(module, my_nn.Conv2dNPU):
                        writer.add_scalar('val/non_zore_num', module.non_zero_num_ins, epoch)
                        break
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
                }, is_best, prefix='{}/{}_w{}a{}'.format(args.log_name, args.arch, args.qw, args.qa))
        else:
            # prune and quantize the model
            wrapper.replace_conv_recursively(model, 'Conv2dSQ', nbits_a=args.qa, nbits_w=args.qw,
                                             sparsity=args.sparsity,
                                             total_iter=args.batch_num * args.epochs, INS=args.INS, beta=args.beta)
            args.arch = '{}_sq'.format(args.arch)
            print(model)
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
