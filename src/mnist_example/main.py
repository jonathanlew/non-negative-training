import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class CustomOptimizer(optim.Optimizer):
    """SGD with momentum and decoupled weight decay.

    If nonneg=True, weight matrices/kernels (dim > 1) are projected onto
    the non-negative orthant (clamped to >= 0) after every step. Biases are
    left unconstrained.
    """
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, nonneg=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nonneg=nonneg)
        super(CustomOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum, wd = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if wd != 0:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.clone(p.grad)
                else:
                    state['momentum_buffer'].mul_(momentum).add_(p.grad)
                p.add_(state['momentum_buffer'], alpha=-lr)
                if group['nonneg'] and p.dim() > 1:
                    p.clamp_(min=0)

        return loss


def init_weights(model, nonneg=False):
    """Kaiming-uniform weights, zero biases.

    If nonneg=True, weights are drawn from U(0, 2/fan_in) so E[w] = 1/fan_in.
    With all-positive weights the activation mean (not the variance) is what
    compounds across layers, so this keeps it roughly constant with depth.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if nonneg:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                nn.init.uniform_(m.weight, 0.0, 2.0 / fan_in)
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 0.01 for custom, 1.0 for adadelta)')
    parser.add_argument('--optimizer', choices=['custom', 'adadelta'], default='custom',
                        help='optimizer to use (default: custom)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum for the custom optimizer (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='WD',
                        help='weight decay for the custom optimizer (default: 0.0)')
    parser.add_argument('--init', choices=['custom', 'default'], default='custom',
                        help='weight initializer (default: custom)')
    parser.add_argument('--nonneg', action='store_true',
                        help='non-negative init and keep weights >= 0 during training')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    if args.init == 'custom':
        init_weights(model, nonneg=args.nonneg)
    model = model.to(device)

    if args.optimizer == 'custom':
        lr = args.lr if args.lr is not None else 0.01
        optimizer = CustomOptimizer(model.parameters(), lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nonneg=args.nonneg)
    else:
        lr = args.lr if args.lr is not None else 1.0
        optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
