import torch

import sys

sys.path.insert(0, './audio_models/ConvNets_SpeechCommands')

def create_model(path):

    model = torch.load(path).module
    model.float()
    model.eval()

    return model