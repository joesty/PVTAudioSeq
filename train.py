from vsum_tools import evaluate_summary, set_summary_from_video_index
from time import sleep
import numpy as np
import pdb
from tqdm import trange
import torch.nn as nn
import torch.optim as optim
#import wandb
from test import inference
import torch

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse((pred + 1), (actual + 1)))        

#def fill_wandb_config(epochs, criterion, lr, weight_decay, optimizer):
#    wandb.config.update({"epochs": epochs,
#                         "criterion": criterion,
#                         "lr": lr,
#                         "weight_decay": weight_decay,
#                         "optimizer": optimizer})

def train(dataloader, model, device, hdf, epochs, lr, weight_decay, eval_method, split, test_dataloader):
    model.to(device)
    model.train()
    # criterion = RMSLELoss()
    criterion = RMSELoss()
    #criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=750, max_iters=7500)
    progress_bar = trange(epochs, desc="Traning Epoch", ncols=80, leave=True)
    # fill_wandb_config(epochs, 
    #                   criterion, 
    #                   lr, 
    #                   weight_decay, 
    #                   optimizer.__class__.__name__)
    #wandb.watch(model)
    for epoch in progress_bar:
        loss = 0
        for frame_features, audio_features, target, patarget, user_summary, gt_summary, video_name in dataloader:
            video_index = video_name[0][6:]
            optimizer.zero_grad()
            # user_summary = target.numpy()
            user_summary = user_summary.numpy()[0]
            outputs, _ = model(frame_features.to(device).squeeze(0),
                               audio_features.to(device).squeeze(0) 
                               #patarget.to(device).squeeze(0))
                            )
            #print(outputs.shape)
            #print(patarget.shape)
            # outputs, _ = model(frame_features.to(device).squeeze(0))
            video_loss = criterion(outputs.cpu().squeeze(0), target.squeeze(0))
            #video_loss = criterion(outputs.cpu().squeeze(0), patarget.squeeze(0))
            # print(video_name, patarget.mean())
            # video_loss = criterion(outputs.cpu().squeeze(0), gt_summary.squeeze(0))
            loss += video_loss

            if epoch % 500 == 0:
                scores = outputs.squeeze(0).cpu().detach().numpy().tolist()
                summary = set_summary_from_video_index(hdf, video_index, scores)
                f1_score, prec, rec = evaluate_summary(summary, user_summary, 'max')
                import matplotlib.pyplot as plt
                #plt.plot(target.cpu().numpy()[0])
               #plt.plot(patarget.cpu().numpy()[0])
                # plt.plot(gt_summary.cpu().numpy()[0])
                #plt.plot(outputs.detach().cpu().numpy()) ; plt.show()

                print("VIDEO LOSS: {}".format(video_loss))
                target_mean = target.cpu().numpy()[0].mean()
                print("GT MEAN: {}".format(target_mean))
                print("Scores MEAN: {}".format(outputs.detach().cpu().numpy().mean()))
                print("F1 Score: {}".format(f1_score))
                print("Precision: {}".format(prec))
                print("Recall: {}".format(rec))
        #if epoch % 5 == 0:
        #    f1_test = inference(model, eval_method, hdf, split, test_dataloader, device)

            # wandb.log({
            #   'train/loss/video_{}'.format(video_index): loss.item(),
            #   'train/f1_score/video_{}'.format(video_index): f1_score,
            #   'train/precision/video_{}'.format(video_index): prec,
            #   'train/recall/video_{}'.format(video_index): rec,
            #   'train/frame_scores/video_{}'.format(video_index): wandb.Histogram(np_histogram=np.histogram(a=range(len(scores)), weights=scores))
            # })
            # print("F1: {}".format(f1_score))
        # loss = loss / len(dataloader)
        loss = loss / len(dataloader)
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        #wandb.log({'train/loss': loss.item()})
        #wandb.log({'test/f1': f1_test})
    return model
