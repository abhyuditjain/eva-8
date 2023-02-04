import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dataloader import Cifar10DataLoader
from utils import imshow
from model import Net
from train import Trainer
from test import Tester
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau


def print_summary(model):
    batch_size = 20
    s = summary(
        model,
        input_size=(batch_size, 3, 32, 32),
        verbose=1,
        col_names=[
            "kernel_size",
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
            "trainable",
        ],
        row_settings=["var_names"],
    )


def main():
    # is_cuda_available = torch.cuda.is_available()
    # cifar10 = Cifar10DataLoader(is_cuda_available=is_cuda_available)
    # classes = cifar10.get_classes()
    # train_loader = cifar10.get_loader(True)
    # train_iter = iter(train_loader)
    # images, labels = next(train_iter)

    # imshow(torchvision.utils.make_grid(images[:4]))
    # print(" ".join("%5s" % classes[labels[j]] for j in range(4)))

    is_cuda_available = torch.cuda.is_available()
    print("Is GPU available?", is_cuda_available)
    device = torch.device("cuda" if is_cuda_available else "cpu")
    cifar10 = Cifar10DataLoader(is_cuda_available=is_cuda_available)
    train_loader = cifar10.get_loader(True)
    test_loader = cifar10.get_loader(False)
    model = Net().to(device=device)

    print_summary(model)

    trainer = Trainer()
    tester = Tester()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.NLLLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=0)

    EPOCHS = 25

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        accuracy = trainer.train(
            model, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step(accuracy)
        tester.test(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
