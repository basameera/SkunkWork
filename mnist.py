"""Project Doc String - Python program tempalte"""
# imports
from skunkwork.pytorchCustomDataset import *
from skunkwork.skunkwork import *
from skunkwork.swTrainer import *
from skunkwork.utils import *

#
from torch.utils.data import DataLoader, random_split
import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
#
import argparse
import time


class customModel(nn.Module):

    def __init__(self, in_channels=1, out_channels=10):

        # Basics
        super(customModel, self).__init__()

        # Initializing all layers
        self.conv1 = nn.Conv2d(in_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(800, 500)  # mnist
        self.fc2 = nn.Linear(500, out_channels)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# custom classes and functions


def cmdArgs():
    parser = argparse.ArgumentParser(
        description='PyTorch NN\n- by Bassandaruwan')
    batch_size = 64
    valid_batch_size = 32
    epochs = 1
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: {})'.format(batch_size))
    parser.add_argument('--valid-batch-size', type=int, default=valid_batch_size, metavar='N',
                        help='input batch size for validating (default: {})'.format(valid_batch_size))
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: {})'.format(epochs))
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the trained Model')
    parser.add_argument('--save-best', action='store_true', default=False,
                        help='For Saving the current Best Model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Start Training the model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Start Evaluating the model')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Load the model')
    parser.add_argument('--show_progress', action='store_true', default=True,
                        help='Show training progress')
    parser.add_argument('--save_plot', action='store_true', default=True,
                        help='Save the loss plot as .png')
    return parser.parse_args()


def main():
    args = cmdArgs()
    prettyPrint(args.__dict__, 'cmd args')
    use_cuda = not args.no_cuda and cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    

    # Pytorch Custom Dataset
    # norm_mean = [0.1349952518939972]
    # norm_std = [0.30401742458343506]
    # data_folder_path = '/home/sameera/Github/image-data/MNIST'
    # input_dataset = ImageClassDatasetFromFolder(
    #     data_folder_path, int_classes=True, norm_data=True, norm_mean=norm_mean, norm_std=norm_std, size=28)
    # print('Classes:', input_dataset.getClasses())
    # print('Decode Classes:', input_dataset.getInvClasses())
    # print('Dataset split radio (train, validation, test):',
    #       getSplitByPercentage(0.8, len(input_dataset)))

    # train_dataset, val_dataset, test_dataset = random_split(
    #     input_dataset, getSplitByPercentage(0.8, len(input_dataset)))

    num_workers = 4

    # Pytorch MINST Dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../pytorch_MNIST_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../pytorch_MNIST_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.valid_batch_size, shuffle=True, **kwargs)

    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=False,
    #                           pin_memory=True,
    #                           num_workers=num_workers)
    # valid_loader = DataLoader(dataset=val_dataset,
    #                           batch_size=args.valid_batch_size,
    #                           shuffle=True,
    #                           pin_memory=True,
    #                           num_workers=num_workers)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=1,
    #                          shuffle=True,
    #                          pin_memory=True,
    #                          num_workers=num_workers)

    clog('Data Loaders ready')

    settings = dict()
    

    # reproducibility
    torch.manual_seed(0)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # settings
    settings['use cuda'] = use_cuda
    settings['device'] = 'cpu' if (not use_cuda) else (
        'cuda:'+str(cuda.current_device()))
    settings['device'] = torch.device(settings['device'])
    settings['in_channels'] = 1
    settings['out_channels'] = 10

    prettyPrint(settings, 'settings')

    clog('Model Ready')

    model = customModel(
        in_channels=settings['in_channels'], out_channels=settings['out_channels'])
    print(model.eval())

    trainer = nnTrainer(model=model, model_name=__file__,
                        use_cuda=settings['use cuda'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer.compile(optimizer, criterion=nn.CrossEntropyLoss(),
                    valid_criterion=nn.CrossEntropyLoss())  # reduction='mean'

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    clog('Model Total Trainable parameters: {}'.format(pytorch_total_params))

    # Train model
    if args.train:
        clog('Training Started...\n')
        start_time = time.time()
        history = trainer.fit(train_loader, test_loader, epochs=args.epochs, save_best=args.save_best,
                              show_progress=args.show_progress, save_plot=args.save_plot)
        clog("Training time: {} seconds | Device: {}".format(
            time.time() - start_time, settings['device']))
        clog('History', history)

    # save model
    if args.train and args.save_model:
        trainer.saveModel(path='model_'+str(args.epochs), full=False)

    if args.eval:
        # test model
        clog('Prediction Test model')
        output = trainer.predict(test_loader, show_progress=True)


# run
if __name__ == '__main__':
    print('\n\n')
    clog(__file__)
    main()
