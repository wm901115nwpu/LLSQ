import torchvision.models as models

from examples import *
# from models._modules.bit_pruning import count_bit
from models._modules.sq import get_sparsity_mask


# def test_log_shift():
#     random_tensor = torch.zeros(2)
#     random_tensor[0] = 1.1
#     random_tensor[1] = 2.2
#     shift = log_shift(random_tensor)
#     assert shift[0] == 1
#     assert shift[1] == 2

#
# def test_count_bit():
#     random_tensor = torch.zeros(2)
#     random_tensor[0] = 1
#     random_tensor[1] = 2
#     bit_cnt = count_bit(random_tensor)
#     assert bit_cnt[0] == 1
#     assert bit_cnt[1] == 1


def test_get_lr_scheduler():
    args = argparse.Namespace()
    args.epochs = 100
    args.lr_scheduler = 'MultiStepLR'
    args.step_size = 20
    args.gamma = 0.1
    args.milestones = [10, 15, 40]
    args.lr = 1
    args.warmup_epoch = -1

    model = models.alexnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9)
    scheduler = get_lr_scheduler(optimizer, args)
    for i in range(args.epochs):
        scheduler.step()
        print('{}: {}'.format(i, optimizer.param_groups[0]['lr']))


def test_get_sparsity_mask():
    param = torch.zeros(4)
    param[0] = 5
    param[1] = 6
    param[2] = 3
    param[3] = 4
    mask = get_sparsity_mask(param, 0.5)
    print(mask)


if __name__ == '__main__':
    test_get_sparsity_mask()
