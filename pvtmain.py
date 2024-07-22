# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import json
import h5py
from tqdm import tqdm, trange
from models.PGL_SUM import PGL_SUM
from data_loader import get_pgl_loader, get_loader
from generate_summary import generate_summary
from evaluate import evaluate_summary
from torch.utils.data.dataloader import DataLoader
from models.PVTAudio import PVTAudioSeq, PVTAudioSeqSeparateMLPs

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse((pred + 1), (actual + 1)))  

class Config:
    """Configuration class for PGL-SUM model."""
    def __init__(self, **kwargs):
        self.mode = 'train'
        self.seed = 12345
        self.device = torch.device('cuda')
        self.verbose = True
        self.log_dir = 'logs'
        self.init_type = 'xavier'
        self.init_gain = None
        self.input_size = 1024
        self.n_segments = 4
        self.heads = 8
        self.fusion = 'add'
        self.pos_enc = 'absolute'
        self.lr = 5e-5
        self.l2_req = 1e-5
        self.clip = 5.0
        self.batch_size = 1
        self.n_epochs = 200

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

class Solver(object):
    def __init__(self, config=None):
        """Class that Builds, Trains and Evaluates PGL-SUM model"""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def run(self, splits=5):
        self.build()
        for i in range(splits):
            self.train_loader = get_loader('train', 'SumMe', i)
            self.test_loader = get_loader('test', 'SumMe', i)
            self.train()

    def build(self):
        """ Function for constructing the PGL-SUM model of its key modules and parameters."""
        # Model creation
        #self.model = PVTAudioSeq(emb_size=self.config.input_size, heads=8, enc_layers=4, dropout=0.5)
        
        self.model = PVTAudioSeqSeparateMLPs(emb_size=self.config.input_size, heads=8, enc_layers=4)
        self.model.to(self.config.device)
        
        if self.config.init_type is not None:
            self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

        if self.config.mode == 'train':
            # Optimizer initialization
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_req)

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    #criterion = nn.MSELoss()
    criterion = RMSELoss()
    def train(self):
        """ Main function to train the PGL-SUM model. """
        for epoch in tqdm(range(0, self.config.n_epochs)):
            self.model.train()
            loss_history = []
            num_batches = int(len(self.train_loader) / self.config.batch_size)
            for frame_features, audio_features, target, pascore, user_summary, gt_summary, video_name, _, _, _, _ in tqdm(self.train_loader, desc='Batch', ncols=80):
                frame_features, target = frame_features.to(self.config.device).squeeze(0), target.to(self.config.device).squeeze(0)
                self.optimizer.zero_grad()
                output, _ = self.model(frame_features.to('cuda'), audio_features.to('cuda').squeeze(0) )
                loss = self.criterion(output.squeeze(0), target)
                loss.backward()
                self.optimizer.step()
                loss_history.append(loss.item())
            if epoch % 50 == 0:
                self.test()
            #print(loss_history)
        self.test()

    def test(self):
        """ Main function to test the PGL-SUM model. """
        self.model.eval()

        loss_history = []
        f1s = []
        for frame_features, audio_features, gtscore, pascore, user_summary, gt_summary, video_name, cps, n_frames, nfps, positions in tqdm(self.test_loader, desc='Batch', ncols=80):
            frame_features = frame_features.to(self.config.device).squeeze(0)
            with torch.no_grad():
                output, _ = self.model(frame_features, audio_features.to(self.config.device).squeeze(0) )
            #print('')
            #print('N Frames:', n_frames.numpy()[0])
            #print('NFPS:', nfps.numpy()[0])
            #print('Positions:', positions.numpy()[0])
            #print('CPS:', cps.numpy()[0])
            cps = cps.numpy()[0]
            n_frames = n_frames.numpy()[0]
            nfps = nfps.numpy()[0]
            positions = positions.numpy()[0]
            output = output.squeeze(0).cpu().detach().numpy().tolist()
            summary = generate_summary(
                output, cps=cps,
                n_frames=n_frames, nfps=nfps,
                positions=positions, proportion=0.15,
                method='knapsack'
            )
            f1, prec, rec = evaluate_summary(summary, user_summary.numpy()[0], 'avg')
            f1s.append(f1)
            print("Video: {}, F1: {}, Precision: {}, Recall: {}".format(
                video_name[0], f1, prec, rec
            ))
        print("Done testing!")
        print("Average F1 Score: ", np.mean(f1s))
if __name__ == '__main__':
    # Configuration of the PGL-SUM model
    config = Config()
    # Initialize the PGL-SUM model
    solver = Solver(config=config)
    # Train the PGL-SUM model
    solver.run()