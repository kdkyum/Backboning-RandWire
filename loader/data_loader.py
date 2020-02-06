import random
import numpy as np
import torch
import os
from torchvision import transforms, datasets

__all__ = ["cifar10", "cifar100", "tiny_imagenet"]

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def cifar10(directory, augmentation=False, cutout=False):
    normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if cutout:
        train_transform.transforms.append(Cutout(16))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.CIFAR10(root=directory, train=True, download=True,
                                transform=train_transform if augmentation else test_transform)
    validset = datasets.CIFAR10(root=directory, train=True, download=True,
                                transform=test_transform)
    testset = datasets.CIFAR10(root=directory, train=False, download=True, transform=test_transform)

    nClasses = 10
    in_shape = [3, 32, 32]

    return trainset, validset, testset, nClasses, in_shape


def cifar100(directory, augmentation=False, cutout=False):
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2761)
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if cutout:
        train_transform.transforms.append(Cutout(8))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.CIFAR100(root=directory, train=True, download=True,
                                 transform=train_transform if augmentation else test_transform)
    validset = datasets.CIFAR100(root=directory, train=True, download=True,
                                 transform=test_transform)
    testset = datasets.CIFAR100(root=directory, train=False, download=True, transform=test_transform)

    nClasses = 100
    in_shape = [3, 32, 32]

    return trainset, validset, testset, nClasses, in_shape


def tiny_imagenet(directory, augmentation=False, cutout=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_dir = os.path.join(directory, 'train')
    val_dir = os.path.join(directory, 'valid')

    if cutout:
        trainset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        trainset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    validset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    nClasses = 200
    in_shape = [3, 64, 64]

    return trainset, trainset, validset, nClasses, in_shape
