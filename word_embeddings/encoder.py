import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from copy import deepcopy
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR, StepLR, MultiStepLR, ConstantLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.nn import Sequential, Linear, BatchNorm1d
from torch.optim import Adam, SGD


sys.path.append(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dim', type=int, default=300,
        help='dimension of the input (original) embeddings')
    parser.add_argument(
        '--hidden-dims', nargs='+', type=int, default=[200, 150],
        help='specify the number of neurons per hidden layer, e.g. 200 100 50 -> enocder will have 3 hidden layers (first layer has 200 neurons...)')
    parser.add_argument(
        '--latent-dim', type=int, default=20,
        help='dimension the word embeddings should get reduced to')
    parser.add_argument(
        '--batch-size', type=int, default=64)
    parser.add_argument(
        '--lr', type=float, default=0.1)
    parser.add_argument(
        '--epochs', type=int, default=100)
    parser.add_argument(
        '--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh', 'gelu', 'selu'],
        help='activation function for hidden layers')
    parser.add_argument(
        '--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument(
        '--lr-scheduler', type=str, default='onecycle', choices=['onecycle', 'step', 'multistep', 'constant', 'linear', 'cosine', 'cosinewarm'])
    parser.add_argument(
        '--ring-loss', action='store_true', default=False,
        help='ring loss achieves a normalization of the reduced word embeddings')
    parser.add_argument(
        '--target-length', type=float, default=1.0,
        help='length of embeddings when using ring loss')
    parser.add_argument(
        '--use-bias', action='store_true', default=False,
        help='whether or not to use bias in hidden layers')
    parser.add_argument(
        '--loss-weighting', nargs='+', type=float, default=[1.0, 1.0],
        help='weighting for cosine loss and ring loss (sequence: cosine loss, ring loss)')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='weight decay for optimizer')
    parser.add_argument(
        '--momentum', type=float, default=0.9,
        help='momentum for optimizer')
    parser.add_argument(
        '--data', type=str, default='./vocabulary.txt',
        help='path to embeddings for training the encoder')
    parser.add_argument(
        '--result-path', type=str, default='./results_encoder/')
    parser.add_argument(
        '--wandb', action='store_true', default=False,
        help='log results with Weights and Biases')

    args = parser.parse_args()
    return args


def load_data(data_path):
    """Load vocabulary of words and corresponding word embeddings for training the encoder.
        Args:
            data_path (str): Path to vocabulary .txt file.

        Returns:
            Embeddings: List of word embeddings.
            Word_to_embedding: Dict for words/embeddings.
    """
    word_to_embedding = {}
    embeddings = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            w = True
            values = line.split()
            word = values[0]
            i = 1
            while w:
                # check if entry is number, otherwise it still belongs to the embedded word(s)
                try:
                    vector = torch.tensor([float(val) for val in values[i:]], dtype=torch.float)
                    w = False
                except ValueError:
                    word += ' '
                    word += values[i]
                    i +=1
            word_to_embedding[word] = vector
            embeddings.append(vector)
    
    embeddings = torch.stack(embeddings)

    return embeddings, word_to_embedding


class EmbeddingDataset(Dataset):
    def __init__(self, data_path):
        self.embeddings, self.word_to_emb = load_data(data_path)

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings[index]


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        activation = get_activation(args.activation)
        modules = []
        input_dim = args.input_dim
        # define layers of encoder
        for h_dim in args.hidden_dims:
            modules.append(
                Sequential(
                    Linear(input_dim, h_dim, bias=args.use_bias),
                    BatchNorm1d(h_dim),
                    activation)
            )
            
            input_dim = h_dim
        
        self.fc_out = Linear(input_dim, args.latent_dim)
        modules.append(self.fc_out)

        self.encoder = Sequential(*modules)

        self.lr = args.lr
        self.optimizer = args.optimizer
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dataset = EmbeddingDataset(args.data)

    def forward(self, x):
        y = self.encoder(x)
        return y
    
    def compute_loss(self, target, output):
        # compute cosine similarity loss
        output_sim = F.cosine_similarity(output.unsqueeze(1), output.unsqueeze(0), dim=-1)
        target_sim = F.cosine_similarity(target.unsqueeze(1), target.unsqueeze(0), dim=-1)
        loss = torch.square(output_sim - target_sim).mean()
        return loss
    
    def ring_loss(self, output, args):
        target = args.target_length
        norm = torch.norm(output, p=2, dim=1)

        loss = torch.mean((norm - target) ** 2)
        return loss
    
    def get_dataset(self):
        return self.dataset
    
    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True)


def get_activation(name):
    # get activation for hidden layers of encoder
    name = name.lower()
    if name not in ['relu', 'sigmoid', 'tanh', 'gelu', 'selu']:
        raise ValueError(F"Unknown activation: '{name}'")
    if 'relu' == name:
        activation = nn.ReLU()
    elif 'sigmoid' == name:
        activation = nn.Sigmoid()
    elif 'tanh' == name:
        activation = nn.Tanh()
    elif 'gelu' == name:
        activation = nn.GELU()
    elif 'selu' == name:
        activation = nn.SELU()
    return activation


def get_optimizer(args, parameters):
    name = args.optimizer
    name = name.lower()
    if name not in ['sgd', 'adam']:
        raise ValueError(f"Unknown optimizer: '{name}'")

    if 'sgd' == name:
        optimizer = SGD(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif 'adam' == name:
        optimizer = Adam(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

    return optimizer


def get_lr_scheduler(args, optimizer):
    name = args.lr_scheduler
    n_epochs = args.epochs

    name = name.lower()
    if name not in ['onecycle', 'step', 'multistep', 'constant', 'linear', 'cosine', 'cosinewarm']:
        raise ValueError(f"Unknown learning rate scheduler: '{name}'")

    if 'onecycle' == name:
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=[i['lr'] for i in optimizer.param_groups],
            total_steps=n_epochs,
            div_factor=25,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e4
        )

    elif 'step' == name:
        lr_scheduler = StepLR(optimizer, step_size=args.epochs/4)

    elif 'multistep' == name:
        lr_scheduler = MultiStepLR(optimizer, milestones=[args.epochs/3, args.epochs/3*1.5, args.epochs/3*2.5])

    elif 'constant' == name:
        lr_scheduler = ConstantLR(optimizer)

    elif 'linear' == name:
        lr_scheduler = LinearLR(optimizer)

    elif 'cosine' == name:
        lr_scheduler = CosineAnnealingLR(optimizer, 4)

    elif 'cosinewarm' == name:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1, 1e-5)

    return lr_scheduler


def main():
    args = parse_args()

    model = Encoder(args)
    if args.wandb:
        args_ = deepcopy(args)
        for k, v in dict(vars(args_)).items():
            if isinstance(v, (list, tuple)):
                v_str = ', '.join(str(v_) for v_ in v)
                if not isinstance(v[0], str):
                    v_str = f's {v_str}'
                setattr(args_, f'{k}_str', v_str)
    
        wandb.init(
        entity='dimension_reduction',
        project='encoder_embeddings',
        config=args_
        )

        args.run_name = wandb.run.name

    # create result directory and dump args to .json file
    now = datetime.now().strftime('%Y_%m_%d-%H_%M_%S-%f')
    path = f'{args.result_path}/{now}/'
    os.makedirs(path)
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    optimizer = get_optimizer(args, model.parameters())
    lr_scheduler = get_lr_scheduler(args, optimizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Start training...")
    best_loss = float('inf')

    # training
    for epoch in range(model.epochs):
        model.train()
        train_loss = ring_loss = cosine_loss = loss = 0

        for idx, batch in enumerate(model.train_dataloader()):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            c_loss = model.compute_loss(batch, output)
            if args.ring_loss:
                r_loss = model.ring_loss(output, args)
                loss = args.loss_weighting[0] * c_loss + args.loss_weighting[1] * r_loss
                ring_loss += r_loss.item()
            else:
                loss = c_loss

            loss.backward()
            train_loss += c_loss.item()
            cosine_loss += c_loss.item()
            optimizer.step()
        lr_scheduler.step()

        train_loss /= (idx + 1) #average loss for better comparability of different vocabulary sizes

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "loss": train_loss,
                "ring_loss": ring_loss,
                "cosine_loss": cosine_loss
            })

        # save model with lowest loss
        if train_loss < best_loss:
            ckpt = {
                'state_dict': model.state_dict(),
                'epoch': epoch
            }
            best_loss = train_loss

    # save checkpoint
    ckpt_path = f'{path}/{ckpt["epoch"]}.pth'
    torch.save(ckpt["state_dict"], ckpt_path)

    # load best checkpoint write reduced embedding of best checkpoint to result file
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        with open(f'{path}/encoder_embeddings.txt', 'w', encoding="utf-8") as f:
            for word, embedding in model.dataset.word_to_emb.items():
                embedding = embedding.unsqueeze(0).to(device)
                reduced_embedding = model(embedding)
                f.write("%s\t" % word)
                for v in reduced_embedding[0].cpu():
                    f.write("%s\t" % str(v.item()))
                f.write("\n")

if __name__ == '__main__':
    main()