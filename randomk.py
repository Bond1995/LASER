#!/usr/bin/env python3

import json
import os
from copy import deepcopy
from typing import Any, Iterable
from dataclasses import dataclass

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
    dataset="Cifar100",
    model="resnet18",
    optimizer="SGD",
    decay_at_epochs=[150, 250],
    decay_with_factor=10.0,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    batch_size=128,
    power=1000,
    num_epochs=200,
    seed=42,
    ema_alpha=0.99,
    algorithm="randomk",
    powersgd_rank=4,
    power_allocation="norm",
    exponent=1,
    workers=16,
    compression_factor=0.2
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
    workers = config["workers"]
    training_loaders, test_loader = get_dataset(workers=workers)
    models = []
    optimizers = []
    schedulers = []
    criterions = []
    models.append(get_model(device))
    for i in range(1,workers):
        models.append(deepcopy(models[0]))
    for i in range(workers):
        opt, sch = get_optimizer(models[i].parameters())
        optimizers.append(opt)
        schedulers.append(sch)
        criterions.append(torch.nn.CrossEntropyLoss())
    params = [list(model.parameters()) for model in models]
    # Generator for Random-K algorithm
    generator = torch.Generator().manual_seed(0)

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()

    mean_test_accuracy = utils.accumulators.Mean()

    start_epoch = 0

    if wandb.run is not None and wandb.run.resumed:
        wandb.restore("checkpoint.pt")
        checkpoint = torch.load("checkpoint.pt")
        for i in range(workers):
            models[i].load_state_dict(checkpoint["model_state_dict"])
            schedulers[i].load_state_dict(checkpoint["scheduler_state_dict"])
            optimizers[i].load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_accuracy_so_far.add(checkpoint["best_accuracy_so_far"])
        print(f"Resuming from epoch {start_epoch}.")

    average_weights = utils.accumulators.EMA(alpha=config["ema_alpha"])
 
    for epoch in range(start_epoch, config["num_epochs"]):
        # Enable training mode (automatic differentiation + batch norm)
        for model in models:
            model.train()
        
        # Keep track of statistics during training
        mean_train_accuracy = utils.accumulators.Mean()
        mean_train_loss = utils.accumulators.Mean()

        assert len(training_loaders[0]) == len(training_loaders[workers-1])
        for batch_set in zip(*training_loaders):
            for i in range(workers):
                batch_x, batch_y = batch_set[i][0].to(device), batch_set[i][1].to(device)
                # Uncomment for no error feedback
                #optimizers[i].zero_grad()
                # Compute gradients for the worker
                prediction = models[i](batch_x)
                loss = criterions[i](prediction, batch_y)
                acc = accuracy(prediction, batch_y)
                loss.backward()
            
            # Do an optimizer step
            if config["algorithm"] == "randomk":
                grads = []
                for i in range(workers):
                    grads.append([p.grad.data for p in params[i]])
                avg_grads = aggregate(params, generator, config["power"])
                # Set average grads
                for p, g in zip(params[0], avg_grads):
                    p.grad = g
                optimizers[0].step()
                # Update model parameters
                for w in params:
                    for i in range(len(w)):
                        w[i].data = params[0][i].data
                # Put back error feedback
                for i in range(workers):
                    for (p, g) in zip(params[i], grads[i]):
                        p.grad = g
            else:
                raise ValueError("Unknown algorithm")

            average_weights.add(models[0].state_dict())

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

        # Update the optimizers' learning rates
        for i in range(workers):
            schedulers[i].step()

        # Log training stats
        wandb.log(
            {
                "train/accuracy": mean_train_accuracy.value(),
                "train/loss": mean_train_loss.value(),
            },
            step=epoch + 1,
        )

        # Evaluation
        models[0].eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = models[0](batch_x)
            loss = criterions[0](prediction, batch_y)
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
        ema_model = deepcopy(models[0])
        ema_model.load_state_dict(average_weights.value())
        ema_model.eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = ema_model(batch_x)
            loss = criterions[0](prediction, batch_y)
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
                "optimizer_state_dict": optimizers[0].state_dict(),
                "scheduler_state_dict": schedulers[0].state_dict(),
                "model_state_dict": models[0].state_dict(),
                "best_accuracy_so_far": best_accuracy_so_far.value(),
            },
            CHECKPOINT_FILENAME,
        )
        wandb.save(CHECKPOINT_FILENAME)

@dataclass
class RandomkCompressed:
    seed: int
    original_shape: torch.Size
    vector: torch.Tensor

def aggregate(params, generator, power):
    if len(params[0]) == 0:
        return []
    gradients = []
    for w in params:
        gradients.append([p.grad.data for p in w])
    device = gradients[0][0].device
    workers = len(gradients)
    out = [torch.zeros_like(g) for g in gradients[0]]
    mean_grads = []
    for g in gradients[0]:
        compressed_size = int(config["compression_factor"] * g.numel())
        if compressed_size == 0:
            compressed_size = 1
        mean_grads.append(torch.zeros(compressed_size).to(device))
    shared = [int(torch.randint(1000000000, (1,), generator=generator)) for g in gradients[0]]
    power_per_param = torch.zeros(len(gradients[0])).to(device)
    noise_coeffs = torch.zeros(len(gradients[0])).to(device)

    for i in range(workers):
        worker_power = torch.zeros(len(gradients[0])).to(device)
        for j, g in enumerate(gradients[i]):
            compressed_size = int(config["compression_factor"] * g.numel())
            if compressed_size == 0:
                compressed_size = 1
            compressed = compress(g, compressed_size, shared[j])
            worker_power[j] = torch.linalg.norm(compressed.vector)
            # Average gradient
            mean_grads[j].add_(compressed.vector, alpha=1/workers)
            # Error feedback
            g.add_(uncompress(compressed), alpha=-1)
        noise_coeffs.copy_(torch.maximum(noise_coeffs, worker_power))
        power_per_param.add_(F.normalize(worker_power, p = 1, dim = 0), alpha=power/workers)

    # Add noise to compressed grads
    for g, p, c in zip(mean_grads, power_per_param, noise_coeffs):
        g.add_(torch.randn_like(g), alpha=c/(workers*torch.sqrt(p)))
    
    # Reconstruct grads
    for i, g in enumerate(mean_grads):
        out[i].add_(uncompress(RandomkCompressed(seed=shared[i], original_shape=out[i].shape, vector=g)))

    return out


def compress(tensor: torch.Tensor, compressed_size: int, seed: int) -> RandomkCompressed:
    device = tensor.device
    flat_tensor = tensor.view(-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randperm(tensor.numel(), generator=generator, device=device)[:compressed_size]
    compressed = torch.zeros(compressed_size, device=device)
    compressed.add_(flat_tensor[indices])

    return RandomkCompressed(seed=seed, original_shape=tensor.shape, vector=compressed)


def uncompress(compressed: RandomkCompressed) -> torch.Tensor:
    device = compressed.vector.device
    uncompressed = torch.zeros(compressed.original_shape.numel(), device=device)
    generator = torch.Generator(device=device).manual_seed(compressed.seed)
    indices = torch.randperm(compressed.original_shape.numel(), generator=generator, device=device)[:compressed.vector.numel()]
    uncompressed[indices] = compressed.vector

    return uncompressed.view(*compressed.original_shape)


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def get_dataset(
    test_batch_size=1000,
    shuffle_train=True,
    data_root=os.getenv("DATA_DIR", "./data"),
    workers=1
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

    training_set = torch.utils.data.Subset(training_set, torch.randperm(len(training_set)))
    len_per_worker = int(len(training_set) / workers)
    training_sets = []
    for i in range(workers):
        training_sets.append(torch.utils.data.Subset(training_set, range(i*len_per_worker, (i+1)*len_per_worker)))

    training_loaders = []
    for i in range(workers):
        training_loaders.append(torch.utils.data.DataLoader(
        training_sets[i],
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
    ))
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False
    )

    return training_loaders, test_loader

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
