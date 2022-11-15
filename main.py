'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import yaml 
import wandb
import argparse

from models import build_model
from utils import progress_bar

def parser_config():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--cfg_file', type=str, default=None, 
                        help='specify the config for training')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--extra_tag', type=str, default='default', 
                        help='extra tag for this experiment')
    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as f:
        config = yaml.safe_load(f)

    return args , config 

def train(args , model , device , train_loader , optimizer , loss_fn , epoch):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device) , target.to(device)
        
        # Reset the gradients to 0 for all leanrnable weight 
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Compute loss
        loss = loss_fn(output , target)
        # Logging loss / accuracy to wandb
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        
        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()
        # Update the neural network weights
        optimizer.step()
    wandb.log({
        "Train Accuracy": 100. * correct / len(train_loader.dataset),
        "Train Loss": train_loss / len(train_loader.dataset)})
        

def test(args , model , device , test_loader , loss_fn , classes , ):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    total = 0 
    
    example_images = []
    with torch.no_grad():
        for data , target in test_loader:
            data , target = data.to(device) , target.to(device)
            
            # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
            output = model(data)
            
            loss = loss_fn(output , target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            
            # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(classes[predicted[0].item()], classes[target[0]])))
    
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss / len(test_loader.dataset)})
    
    
def main():
    # WandB – Initialize a new run
    wandb.init( project="cifar10-model")
    
    args , cfg = parser_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config 
    
    config.class_name = cfg['CLASS_NAMES']
    config.batch_size = cfg['OPTIMIZATION']['BATCH_SIZE']              # input batch size for training (default: 64)
    config.test_batch_size = cfg['OPTIMIZATION']['TEST_BATCH_SIZE']    # input batch size for testing (default: 1000)
    config.epochs = cfg['OPTIMIZATION']['NUM_EPOCHS']                  # number of epochs to train (default: 10)
    config.lr = cfg['OPTIMIZATION']['LR']                              # learning rate (default: 0.01)
    config.momentum = cfg['OPTIMIZATION']['MOMENTUM']                  # SGD momentum (default: 0.5) 
    config.weight_decay = cfg['OPTIMIZATION']['WEIGHT_DECAY']
    config.no_cuda = False         # disables CUDA training
    config.seed = 42               # random seed (default: 42)
    config.log_interval = 10       # how many batches to wait before logging training status
    
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    # Data
    print('====> Preparing data..')
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('====> Building model..')
    model = build_model(cfg['MODEL'])
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('====> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        # best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    wandb.watch(model, log="all")
    print("====> Start training")
    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, loss_fn , epoch)
        test(config, model, device, test_loader, loss_fn, classes)
        scheduler.step()
    print("====> Saving model")
    # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')

if __name__ == "__main__":
    main()