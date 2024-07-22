from vsum_tools import evaluate_summary, set_summary_from_video_index, generate_summary
from time import sleep

import torch

import numpy as np
import pdb

import matplotlib.pyplot as plt

def test(dataloader, model, device, hdf, print_individual=True, print_metrics=["f1"]):
    model.to(device)
    model.eval()
    f1s = []
    precs = []
    recs = []
    data_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(len(dataloader)):
            frame_features, audio_features, target, patarget, user_summary, gt_summary, video_name = next(data_iter)
            # user_summary = target.numpy()
            
            user_summary = user_summary.numpy()[0]
            outputs, _ = model(frame_features.to(device).squeeze(0),
                               audio_features.to(device).squeeze(0)
                               #patarget.to(device).squeeze(0)
                            )
            # outputs, _ = model(frame_features.to(device).squeeze(0))
            video_index = video_name[0][6:]
            real_video_name = hdf.get('video_' + video_index + '/video_name')[...]
            scores = outputs.squeeze(0).cpu().detach().numpy().tolist()
            summary = set_summary_from_video_index(hdf, video_index, scores)
            f1_score, precision, recall = evaluate_summary(summary, user_summary, 'max')
            if print_individual:
                print("F1 ({}): {}".format(real_video_name, f1_score))
            f1s.append(f1_score)
            precs.append(precision)
            recs.append(recall)
    if "f1" in print_metrics:
        print("Avg F1: {}".format(np.mean(f1s)))
    if "prec" in print_metrics:
        print("Avg Precision: {}".format(np.mean(precs)))
    if "rec" in print_metrics:
        print("Avg Recall: {}".format(np.mean(recs)))


def inference(model, eval_method, hdf, split_id, dataloader, device):
    """ Used to inference a pretrained `model` on the `keys` test videos, based on the `eval_method` criterion; using
        the dataset located in `data_path'.

        :param nn.Module model: Pretrained model to be inferenced.
        :param str data_path: File path for the dataset in use.
        :param list keys: Containing the test video keys of the used data split.
        :param str eval_method: The evaluation method in use {SumMe: max, TVSum: avg}
    """
    model.eval()
    model.to(device)
    video_fscores = []
    for frame_features, audio_features, target, patarget, user_summary, gt_summary, video in dataloader:
        # Input features for inference
        video = video[0]
        # Input need for evaluation
        user_summary = np.array(hdf[f"{video}/user_summary"])
        cps = np.array(hdf[f"{video}/change_points"])
        n_frames = np.array(hdf[f"{video}/n_frames"])
        positions = np.array(hdf[f"{video}/picks"])
        nfps = np.array(hdf[f"{video}/n_frame_per_seg"])
        
        with torch.no_grad():
            scores, _ = model(frame_features.to(device).squeeze(0),
                               audio_features.to(device).squeeze(0)) 
                              #  patarget.to(device).squeeze(0))
            # scores, _ = model(frame_features.to(device).squeeze(0))         
            scores = scores.squeeze(0).cpu().numpy().tolist()
            # plt.xlim([0,700])
            # plt.plot(target.cpu().numpy()[0], color="g")
            # plt.plot(scores, color="deeppink") ; plt.show()
            summary = generate_summary(scores, cps, n_frames, nfps, positions)
            f_score, prec, rec = evaluate_summary(summary, user_summary, eval_method)
            if f_score > 0.1:
                target = target.cpu().numpy()[0]
                #patarget = patarget.cpu().numpy()[0]
                target = np.repeat(target, (len(summary) // len(target)) + 1)[:len(summary)]
                #plt.plot(target, color="y")
                #plt.plot(patarget, color="b")
                #plt.fill_between(x=np.arange(len(target)), y1=target, where=summary, color='cornflowerblue', alpha=0.2)
                #plt.show()

            # print(f_score)
            video_fscores.append(f_score)
            print("Split {}, Video: {}, F1: {}".format(split_id, video, f_score))

    print(f"Trained for split: {split_id} achieved an F-score of {np.mean(video_fscores)}")
    # model.train()
    return np.mean(video_fscores)
