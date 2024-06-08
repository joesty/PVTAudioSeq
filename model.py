import torch
import torch.nn as nn
from models.PGL_SUM import PGL_SUM
from models.PVT import PVT
from models.PVTAudio import PVTAudio, PVTAudioSeq, PVTAudioSeqPA, PVTAudioSeqSeparateMLPs

class RandModel(nn.Module):
    def __init__(self):
        super(RandModel, self).__init__()

    def forward(self, x):
        return torch.rand(x.shape[-2]), None

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.squeeze(1), None

def get_model(model_name, config=None):
    """
    Returns the model class instance.
    :param str model_name: The name of the model.
    :return: The model class instance.
    """
    if model_name == "rand":
        return RandModel()
    elif model_name == "MLP":
        return MLPModel(512, 2048, 1)
    elif model_name == "PGL-SUM":
        return PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                fusion="add", pos_enc="absolute")
    elif model_name == "PVT":
        return PVT(emb_size=512, heads=8, enc_layers=4, 
                 dropout=0.3)
    elif model_name == "PVTAudio":
        return PVTAudio(emb_size=512, heads=8, enc_layers=4, 
                 dropout=0.3)
    elif model_name == "PVTAudioSeq":
        return PVTAudioSeq(emb_size=512, heads=8, enc_layers=4, 
                 dropout=0.3)
    elif model_name == "PVTAudioSeqPA":
        return PVTAudioSeqPA(emb_size=512, heads=8, enc_layers=4, 
                 dropout=0.3)
    elif model_name == "PVTAudioSeqSeparateMLPs":
        return PVTAudioSeqSeparateMLPs(emb_size=512, heads=8, enc_layers=4, 
                 dropout=0.3)
    else:
        raise ValueError("Unknown model name: %s" % model_name)

