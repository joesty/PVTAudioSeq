import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

class PGLData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = ['datasets/eccv16_dataset_summe_google_audio_pa_pool5.h5',
                         'datasets/eccv16_dataset_tvsum_google_pool5.h5']
        self.splits_filename = ['datasets/splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features, self.list_gtscores, self.list_user_summary, self.list_gtsummary = [], [], [], []
        self.list_cps, self.list_n_frames, self.list_positions, self.list_nfps = [], [], [], []
        #self.list_pascores = []
        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
            gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
            user_summary = torch.Tensor(np.array(hdf[video_name + '/user_summary']))
            gt_summary = torch.Tensor(np.array(hdf[video_name + '/gtsummary']))
            #pascore = torch.Tensor(np.array(hdf[video_name + '/pascore']))
            
            cps = np.array(hdf[video_name + '/change_points'])
            n_frames = np.array(hdf[video_name + '/n_frames'])
            positions = np.array(hdf[video_name + '/picks'])
            nfps = np.array(hdf[video_name + '/n_frame_per_seg'])
            
            self.list_frame_features.append(frame_features)
            self.list_gtscores.append(gtscore)
            self.list_user_summary.append(user_summary)
            self.list_gtsummary.append(gt_summary)
            self.list_cps.append(cps)
            self.list_n_frames.append(n_frames)
            self.list_positions.append(positions)
            self.list_nfps.append(nfps)
            #self.list_pascores.append(pascore)

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]
        user_summary = self.list_user_summary[index]
        gt_summary = self.list_gtsummary[index]
        cps = self.list_cps[index]
        n_frames = self.list_n_frames[index]
        positions = self.list_positions[index]
        nfps = self.list_nfps[index]
        #pascore = self.list_pascores[index]


        if self.mode == 'test':
            return frame_features, video_name, user_summary, gt_summary, cps, n_frames, nfps, positions
        else:
            return frame_features, gtscore
        

class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.
        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        # self.datasets = ['datasets/eccv16_dataset_summe_google_pool5.h5', 
        #                  'datasets/eccv16_dataset_tvsum_google_pool5.h5']
        # self.datasets = ['datasets/eccv16_dataset_summe_clip_pool5.h5', 
        #                  'datasets/eccv16_dataset_tvsum_clip_pool5.h5']
        # self.datasets = ['datasets/eccv16_dataset_summe_clip_audio_pool5.h5', 
        #                  'datasets/eccv16_dataset_tvsum_clip_audio_pool5.h5']
        #self.datasets = ['datasets/eccv16_dataset_summe_clip_audio_pa_pool5.h5', 
        #                 'datasets/eccv16_dataset_tvsum_clip_audio_pa_pool5.h5']
        self.datasets = ['datasets/eccv16_dataset_summe_google_audio_pa_pool5.h5', 
                         'datasets/eccv16_dataset_tvsum_google_audio_pa_pool5.h5']
        self.splits_filename = ['datasets/splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features, self.list_gtscores = [], []
        self.list_pascores = []
        self.list_user_summary, self.list_gt_summary = [], []
        self.list_audio_features = []
        self.list_cps, self.list_n_frames, self.list_positions, self.list_nfps = [], [], [], []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
            audio_features = torch.Tensor(np.array(hdf[video_name + '/audio_features']))
            gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
            pascore = torch.Tensor(np.array(hdf[video_name + '/pascore']))
            user_summary = torch.Tensor(np.array(hdf[video_name + '/user_summary']))
            gt_summary = torch.Tensor(np.array(hdf[video_name + '/gtsummary']))
            self.list_frame_features.append(frame_features)
            self.list_gtscores.append(gtscore)
            self.list_pascores.append(pascore)
            self.list_user_summary.append(user_summary)
            self.list_gt_summary.append(gt_summary)
            self.list_audio_features.append(audio_features)

            cps = np.array(hdf[video_name + '/change_points'])
            n_frames = np.array(hdf[video_name + '/n_frames'])
            positions = np.array(hdf[video_name + '/picks'])
            nfps = np.array(hdf[video_name + '/n_frame_per_seg'])

            self.list_cps.append(cps)
            self.list_n_frames.append(n_frames)
            self.list_positions.append(positions)
            self.list_nfps.append(nfps)


        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name
        :param int index: The above-mentioned id of the data.
        """
        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        audio_features = self.list_audio_features[index]
        gtscore = self.list_gtscores[index]
        pascore = self.list_pascores[index]
        user_summary = self.list_user_summary[index]
        gt_summary = self.list_gt_summary[index]
        cps = self.list_cps[index]
        n_frames = self.list_n_frames[index]
        positions = self.list_positions[index]
        nfps = self.list_nfps[index]

        #  if self.mode == 'test':
        #      return frame_features, video_name
        #  else:
        #      return frame_features, gtscore
        return frame_features, audio_features, gtscore, pascore, user_summary, gt_summary, video_name, cps, n_frames, nfps, positions
        


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.
    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=False)
    

def get_pgl_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.
    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = PGLData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        vd = PGLData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=False)


if __name__ == '__main__':
    # dataloader = get_loader('train', 'SumMe', 0)
    # for i, (frame_features, gtscore, video_name) in enumerate(dataloader):
    #     print(frame_features.shape, gtscore.shape, video_name)
    #     if i == 10:
    #         break
    pass