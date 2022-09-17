import torch

import sys

sys.path.insert(0, './audio_models/ConvNets_SpeechCommands')
sys.path.insert(0, './audio_models/M5')

def create_model(path):

    if 'ConvNets_SpeechCommands' in path:
        model = torch.load(path).module
    else:
        model = torch.load(path)
    model.float()
    model.eval()

    return model