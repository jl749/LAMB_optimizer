"""MNIST example.
Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import alexnet, AlexNet_Weights
from tqdm import tqdm

from custom_optimizer import Lamb

base_dir = Path(__file__).parent.parent
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_Train_Test_loaders(where_to_download: Path = base_dir, **kwargs):
    mnist_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # MNIS 1d
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(where_to_download / 'data_mnist', train=True, download=True, transform=mnist_transformer),
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(where_to_download / 'data_mnist', train=False, download=False, transform=mnist_transformer),
        shuffle=True, **kwargs)

    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'adam'],
                        help='which optimizer to use')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--wd', type=float, default=0.01, metavar='WD',
                        help='weight decay (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')

    args = parser.parse_args()

    train_loader, test_loader = get_Train_Test_loaders()

    model = alexnet(AlexNet_Weights.DEFAULT)  # load pre-trained AlexNet
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=2)
    model.classifier[-1] = nn.Linear(4096, 10)

    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(.9, .999),
                     adam=(args.optimizer == 'adam'))
    losses_per_epoch = []
    for epoch in range(args.epochs):
        model.train()
        tqdm_bar = tqdm(train_loader, file=sys.stdout)
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tqdm_bar.set_description(f'Train epoch {epoch} Loss: {loss.item():.6f}')
        losses_per_epoch.append(epoch_loss / len(tqdm_bar))


if __name__ == '__main__':
    main()
