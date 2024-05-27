#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse
import pickle 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

from dynamic_noise_generator import generate_dynamic_noise 


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

all_results={}
eps_acc_results={}

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total=0
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(train_loader.dataset)

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        if epoch%2==0:
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                 f"Accuracy: {accuracy:.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta})"
            )
        return {'epsilon': epsilon, 'acc': accuracy}
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
        return {'epsilon': epsilon, 'acc': accuracy}

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def calculate_averages(data, num_runs):

    aggregate_epsilon = {}
    aggregate_acc = {}
    
    for run in data.values():
        for key, value in run.items():
            if key not in aggregate_epsilon:
                aggregate_epsilon[key] = {'epsilon': 0}
                aggregate_acc[key]={'acc':0}
            aggregate_epsilon[key]['epsilon'] += value['epsilon']
            aggregate_acc[key]['acc'] += value['acc']
    
    # Calculate the averages
    averages_epsilon = {key: value['epsilon'] / num_runs for key, value in aggregate_epsilon.items()}
    averages_acc = {key: value['acc'] / num_runs for key, value in aggregate_acc.items()}
    return {'epsilon':averages_epsilon, 'acc': averages_acc}



def main():
    # Training settings
    np.random.seed(42)


    noise = generate_dynamic_noise('optimized_usefulness/lmo_eps3.json')
    noise = np.abs(noise[0])

    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.2,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=1.5,
        help="The maximum epsilon to be spent",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []
    for run in range(args.n_runs):
        model = SampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None
        eps_acc_results={}
        if not args.disable_dp:
            privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        for epoch in range(1, args.epochs + 1):
            data=train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
            eps_acc_results.update({epoch:data})
            if args.budget > 0 and data['epsilon'] >= args.budget :
                
                break 
            
        all_results.update({run: eps_acc_results})
        run_results.append(test(model, device, test_loader))
        
    average_data= calculate_averages(all_results, args.n_runs)
    file_name=f'data_r2dp_{args.budget}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(average_data, file)

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"mnist_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()
