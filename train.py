#!/usr/bin/env python3 
import os 
import time 
import yaml
import logging 
import argparse
from easydict import EasyDict
from collections import OrderedDict

import torch 
from torch import nn,optim

import utils
from models import build_model

import torchvision
import torchvision.transforms as transforms

_logger = logging.getLogger('train')


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--cfg_file', type=str, default=None, 
                        help='specify the config for training')
    parser.add_argument('-li', '--log_interval', type=int, default=10, 
                        help='screen logs with fix interval (default:10)')
    parser.add_argument('-ne', '--num_epochs', type=int, default=300, 
                        help='number of epochs to train (default: 300)')
    parser.add_argument('-tb', '--train-batch-size', type=int, default=128, 
                        help='Input batch size for training (default: 128)')
    parser.add_argument('-vb', '--vali-batch-size', type=int, default=None, 
                        help='Validation batch size override (default: None)')
    
    # parser.add_argument('--extra_tag', type=str, default='default', 
    #                     help='extra tag for this experiment')
    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    
    return args , config 

def main():
    utils.setup_default_logging()
    args , config  = _parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device 
    device = torch.device(device)

    _logger.info(f'Training with a single process on 1 device ({args.device}).')

    utils.random_seed()
    model = build_model(config.MODEL)
    _logger.info(
        f'Model {config.MODEL} created, param count:{sum([m.numel() for m in model.parameters()])}')
    model.to(device=device)

    # optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=config.OPTIMIZER))
    optimizer = optim.SGD(model.parameters(), lr=config.OPTIMIZATION.LR,
                          momentum=config.OPTIMIZATION.MOMENTUM, weight_decay=config.OPTIMIZATION.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()


    # get dateset
    dataset_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, 
        transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        )

    dataset_eval = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, 
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        )

    # create dataloader
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=args.vali_batch_size, shuffle=False, num_workers=2)

    # setup learning rate schedule
    updates_per_epoch = len(loader_train) 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_batch_size)
    # starting epoch
    start_epoch = 0
    _logger.info(
    f'Scheduled epochs: {args.num_epochs}.')

    for epoch in range(start_epoch , args.num_epochs):
        train_metrics = train_one_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            loss_fn,
            args,
            device=device,
            lr_scheduler=lr_scheduler,
        )

        # eval_metrics = validate(
        #     model,
        #     loader_eval,
        #     loss_fn,
        #     args,
        # )

def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
):
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    end = time.time()
    num_batches_per_epoch = len(loader) #391
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch
    for batch_idx, (input , target) in enumerate(loader):
        last_batch = batch_idx == last_idx 
        data_time_m.update(time.time() - end)

        input, target = input.to(device), target.to(device)
        
        output = model(input)
        loss = loss_fn(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)


            _logger.info(
                'Train: {}/{} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,args.num_epochs,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0)  / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m)
                )

        # if lr_scheduler is not None:
        #     lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time() # end for loop
    return OrderedDict([('loss', losses_m.avg)])
    
if __name__ == '__main__':
    main()