from vsum_tools import generate_summary, evaluate_summary
from knapsack import knapsack_ortools
import h5py
import numpy as np
from natsort import natsorted

dataset = h5py.File("datasets/eccv16_dataset_summe_google_pool5.h5")
for key in natsorted(dataset.keys()):
    #print(key)
    video = dataset[key]
    cps = video["change_points"][...]
    n_frames = video["n_frames"][...]
    nfps = video["n_frame_per_seg"][...]
    positions = video["picks"][...]

    user_summary = video["user_summary"][...]
    # ypred = np.random.rand(n_frames)
    ypred = video["gtsummary"][...]

    machine_summary = generate_summary(ypred, cps, n_frames, nfps, positions)
    # machine_summary = video["gtsummary"][...]

    # print(ypred, machine_summary)

    final_f_score, final_prec, final_rec = evaluate_summary(machine_summary, user_summary, eval_metric='max')
    #print(final_f_score, final_prec, final_rec)
