import argparse
from load_data import load_EOD_data


import torch

from models import HGAM
from Optim import ScheduledOptim
from tool import prepare_dataloaders, train
import torch.optim as optim
import os

import numpy as np



def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()


    parser.add_argument('-length', default=20,
                        help='length of historical sequence for feature')
    parser.add_argument('-train_index', type=int, default=1361)
    parser.add_argument('-valid_index', type=int, default=1531)
    parser.add_argument('-feature', default=10, help='input_size')
    parser.add_argument('-epoch', type=int, default=800)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('--rnn_unit', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--hidden', type=int, default=4, help='Number of hidden units.')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-d_model', type=int, default=32)
    parser.add_argument('-log', default='../days/HGAM')
    parser.add_argument('-save_model', default='../days/HGAM')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    args = parser.parse_args()

    args.cuda = not args.no_cuda

    eod_data, ground_truth = load_EOD_data()
    train_loader, valid_loader = prepare_dataloaders(eod_data, ground_truth, args)#train_loader[1361,758,20,10]

    # ========= Preparing Model =========#
    device = torch.device('cuda' if args.cuda else 'cpu')

    model = HGAM(
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        feature=args.feature,
        dropout=args.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-07),
        args.d_model, args.n_warmup_steps)

    train(model, train_loader, valid_loader, optimizer, device, args)

if __name__ == '__main__':
    main()