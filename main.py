from data_loader import get_loader
from model import get_model
from tqdm import trange

from vsum_tools import evaluate_summary, set_summary_from_video_index
from time import sleep

import torch

import h5py
import numpy as np
import pdb

from test import test, inference
from train import train

from os.path import join, isfile
from os import listdir

import random

#import wandb

from argparse import ArgumentParser

#wandb.init(project="reimplementation", entity="edsonroteia")

seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def main(model_type, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    datasets = ["SumMe", "TVSum"]
    eval_method = {"SumMe": "max", "TVSum": "avg"}
    model_type = model_type #"rand", "MLP", "PVT", "PGL-SUM"
    
    #wandb.config.update({"model_type": model_type,
    #                     "epochs": epochs,
    #                     "lr": lr})

    for dataset in datasets[:1]:
        hdf = h5py.File("datasets/eccv16_dataset_{}_google_pool5.h5".format(dataset.lower()))
 
        split_fscores = []
        num_splits = 5
        for i in range(num_splits):
            if model_type == "PGL-SUM":
                model = get_model(model_type)
                model_path = f"pretrained_models/PGL-SUM/table{3}_models/{dataset}/split{i}"
                model_file = [f for f in listdir(model_path) if isfile(join(model_path, f))]
                model.load_state_dict(torch.load(join(model_path, model_file[-1]), map_location=torch.device('cpu')))
                print("[{}] Training on split{}: ".format(dataset, i))
                train_dataloader = get_loader('train', dataset, i)
                model = train(train_dataloader, model, device, hdf, epochs=epochs, lr=lr, weight_decay=1e-6,
                              eval_method=eval_method[dataset], split=i, test_dataloader=test_dataloader)
                print("[{}] Final train set evaluation: ".format(dataset))
                test(train_dataloader, model, device, hdf)
            elif model_type == "rand":
                model = get_model(model_type)
            else:       
                model = get_model(model_type)
                print("[{}] Training on split{}: ".format(dataset, i))
                train_dataloader = get_loader('train', dataset, i)
                test_dataloader = get_loader('test', dataset, i)
                model = train(train_dataloader, model, device, hdf, epochs=epochs, lr=lr, weight_decay=1e-6,
                              eval_method=eval_method[dataset], split=i, test_dataloader=test_dataloader)
                print("[{}] Final train set evaluation: ".format(dataset))
                test(train_dataloader, model, device, hdf, True)

            print("[{}] Testing on split{}: ".format(dataset, i))
            test_dataloader = get_loader('test', dataset, i)

            
            # test(test_dataloader, model, device, hdf)
            split_fscores.append(inference(model, eval_method[dataset], hdf, i, test_dataloader, device))
            #Average F1 among splits
            avg_f1 = np.mean(split_fscores)
            print("[{}] Average F1: {}".format(dataset, avg_f1))





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_type', default="PVT", type=str)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    
    args = parser.parse_args()
    main(**vars(args))