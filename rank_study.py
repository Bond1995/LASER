#!/usr/bin/env python3

import json
import os
from copy import deepcopy
from typing import Any, Iterable

import models
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends
import torch.utils.data
import torchvision
import utils.accumulators
import wandb
import powersgd

CHECKPOINT_FILENAME = "checkpoint.pt"

config: dict[str, Any] = dict(
    dataset="Cifar10",
    model="resnet18",
    optimizer="SGD",
    decay_at_epochs=[150, 250],
    decay_with_factor=10.0,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    batch_size=1024,
    power=0,
    num_epochs=150,
    seed=42,
    ema_alpha=0.99,
    algorithm="sgd",
    powersgd_rank=8,
    power_allocation="norm",
    exponent=1,
    compressed_only=0
)

# Override config values from environment variables:
for key, default_value in config.items():
    if os.getenv(key) is not None:
        if not isinstance(default_value, str):
            try:
                config[key] = json.loads(os.getenv(key, ""))
            except json.decoder.JSONDecodeError:
                print(f"Failed to decode {key} from environment variable: {os.getenv(key)}.")
                exit(1)
        else:
            config[key] = os.getenv(key, "")


def main():
    wandb.init(config=config, resume="allow")

    # Set the seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # We will run on CUDA if there is a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = get_dataset()
    model = get_model(device)
    optimizer, scheduler = get_optimizer(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Create dict for rank study
    names = ["layer1.0.conv1.weight", "layer3.0.conv2.weight", "layer4.1.conv2.weight"]
    energies = dict() # Energy in Top-8 SV
    for name in names:
        energies[name] = 0.0
    accs = {}
    for name in names:
        accs[name] = utils.accumulators.EMA(alpha=config["ema_alpha"])

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()

    mean_test_accuracy = utils.accumulators.Mean()

    start_epoch = 0

    if wandb.run is not None and wandb.run.resumed:
        wandb.restore("checkpoint.pt")
        checkpoint = torch.load("checkpoint.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_accuracy_so_far.add(checkpoint["best_accuracy_so_far"])
        print(f"Resuming from epoch {start_epoch}.")

    average_weights = utils.accumulators.EMA(alpha=config["ema_alpha"])
    if config["algorithm"] == "powersgd":
        powersgd_config = powersgd.Config(rank=config["powersgd_rank"], power = config["power"], exponent = config["exponent"])
        psgd = powersgd.PowerSGD(list(model.parameters()), powersgd_config)

    for epoch in range(start_epoch, config["num_epochs"]):
        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Keep track of statistics during training
        mean_train_accuracy = utils.accumulators.Mean()
        mean_train_loss = utils.accumulators.Mean()

        for batch_x, batch_y in training_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Compute gradients for the batch
            if config["algorithm"] == "sgd":
                optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            loss.backward()

            # Do an optimizer step
            if config["algorithm"] == "sgd":
                # Apply channel corruption
                if config["power"] > 0:
                    apply_channel([p.grad for p in model.parameters()], config["power"], config["power_allocation"])
                optimizer.step()
            elif config["algorithm"] == "powersgd":
                powersgd.optimizer_step(optimizer, psgd)
            else:
                raise ValueError("Unknown algorithm")

            average_weights.add(model.state_dict())

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

        # Update the optimizer's learning rate
        scheduler.step()

        # Rank study
        for name, p in model.named_parameters():
            if name in names:
                mat = powersgd.view_as_matrix(p.grad.data)
                S = torch.square(torch.linalg.svdvals(mat, driver='gesvd'))
                energy = torch.sum(torch.topk(S, 8)[0]).item() / torch.sum(S).item()
                accs[name].add(energy)
                energies[name] = accs[name].value()

        # Log training stats
        wandb.log(
            {
                "train/accuracy": mean_train_accuracy.value(),
                "train/loss": mean_train_loss.value(),
            },
            step=epoch + 1,
        )

        # Log rank stats
        wandb.log(energies, step=epoch + 1)

        # Evaluation
        model.eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        wandb.log(
            {
                "test/accuracy": mean_test_accuracy.value(),
                "test/loss": mean_test_loss.value(),
            },
            step=epoch + 1,
        )

        # EMA model evaluation
        ema_model = deepcopy(model)
        ema_model.load_state_dict(average_weights.value())
        ema_model.eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = ema_model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        wandb.log(
            {
                "test/ema_accuracy": mean_test_accuracy.value(),
                "test/ema_loss": mean_test_loss.value(),
            },
            step=epoch + 1,
        )

        best_accuracy_so_far.add(mean_test_accuracy.value())
        wandb.summary["best_accuracy"] = best_accuracy_so_far.value()

        torch.save(
            {
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_state_dict": model.state_dict(),
                "best_accuracy_so_far": best_accuracy_so_far.value(),
            },
            CHECKPOINT_FILENAME,
        )
        wandb.save(CHECKPOINT_FILENAME)


def apply_channel(grads: Iterable[torch.Tensor], power: float, power_allocation: str):
    """Apply noise to gradients, in place."""
    grads = list(grads)
    num_grads = len(grads)

    if power_allocation == "uniform":
        power_per_grad = torch.ones(num_grads) * power / num_grads
    elif power_allocation == "param":
        power_per_grad = F.normalize(torch.Tensor([torch.numel(grad) for grad in grads]).float(), p = 1, dim = 0) * power
    elif power_allocation == "param2":
        power_per_grad = F.normalize(torch.Tensor([torch.numel(grad) ** 2 for grad in grads]).float(), p = 1, dim = 0) * power
    elif power_allocation == "norm":
        power_per_grad = F.normalize(torch.Tensor([torch.linalg.norm(grad) for grad in grads]), p = 1, dim = 0) * power
    elif power_allocation == "norm2":
        power_per_grad = F.normalize(torch.Tensor([torch.linalg.norm(grad) ** 2 for grad in grads]), p = 1, dim = 0) * power
    else:
        raise ValueError("Unknown power allocation")
    
    for idx, grad in enumerate(grads):
        channel(grad, power_per_grad[idx].item())

def channel(grad: torch.Tensor, power: float):
    # Apply noise in place
    if power > 0:
        grad.add_(
            torch.randn_like(grad),
            alpha=torch.linalg.norm(grad) / np.sqrt(power)
        )


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def get_dataset(
    test_batch_size=1000,
    shuffle_train=True,
    num_workers=3,
    data_root=os.getenv("DATA_DIR", "./data"),
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    dataset = config["dataset"]
    if dataset == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif dataset == "Cifar100":
        dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Unexpected value for config[dataset] {}".format(dataset))

    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = dataset(
        root=data_root, train=False, download=True, transform=transform_test
    )

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    return training_loader, test_loader


def get_optimizer(model_parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError("Unexpected value for optimizer")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["decay_at_epochs"],
        gamma=1.0 / config["decay_with_factor"],
    )

    return optimizer, scheduler


def get_model(device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = 100 if config["dataset"] == "Cifar100" else 10

    model = {
        "vgg11": lambda: models.vgg11(),
        "vgg13": lambda: models.vgg13(),
        "vgg16": lambda: models.vgg16(),
        "vgg19": lambda: models.vgg19(),
        "efficientnet_b0": lambda: torchvision.models.efficientnet_b0(
            num_classes=num_classes
        ),
        "resnet18": lambda: models.ResNet18(num_classes=num_classes),
        "resnet34": lambda: models.ResNet34(num_classes=num_classes),
        "resnet50": lambda: models.ResNet50(num_classes=num_classes),
        "resnet101": lambda: models.ResNet101(num_classes=num_classes),
        "resnet152": lambda: models.ResNet152(num_classes=num_classes),
    }[config["model"]]()

    model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)

    return model

if __name__ == "__main__":
    main()
