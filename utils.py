import torch
import torchaudio
import numpy as np

import librosa.display
import matplotlib.pyplot as plt
from typing import Union
import os

def spec_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    assert x.shape == (32, 32)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=x, 
                                   x_axis='ms', y_axis='mel', 
                                   sr=16000, n_fft=2048, 
                                   fmin=0, fmax=8000, 
                                   ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    if path is None:
        path = './_Spec_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'spec.png'
    fig.savefig(os.path.join(path, name))

def audio_save(x: Union[np.ndarray, torch.Tensor], path=None, name=None):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu()
    assert x.ndim == 2 and x.shape[0] == 1

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'audio.wav'

    torchaudio.save(os.path.join(path,name), x, 16000) # default sample rate = 16000

def audio_save_as_img(x: Union[np.ndarray, torch.Tensor], path=None, name=None, color=None):
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze()
    assert x.ndim == 1

    fig = plt.figure(figsize=(21, 9), dpi=100)

    from scipy.interpolate import make_interp_spline

    # x_smooth = make_interp_spline(np.arange(0, len(x)), x)(np.linspace(0, len(x), 1000))
    # plt.ylim(-1.5*max(abs(x.max()), abs(x.min())),1.5*max(abs(x.max()), abs(x.min())))
    # plt.plot((np.linspace(0, len(x), 1000)), x_smooth,'-')
    # plt.ylim(-1,1)
    plt.plot(x,'-',color=color if color is not None else 'steelblue', transparent=True)

    if path is None:
        path = './_Audio_Samples'
    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        name = 'waveform.png'

    fig.savefig(os.path.join(path, name))

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), sharey=True)
# p1 = librosa.display.specshow(batch[0].squeeze().numpy(), 
#                                 x_axis='ms', y_axis='mel', 
#                                 sr=16000, n_fft=2048, 
#                                 fmin=0, fmax=8000, 
#                                 ax=ax1, cmap='magma')
# p2 = librosa.display.specshow(batch_purified[0].detach().cpu().squeeze().numpy(), 
#                                 x_axis='ms', y_axis='mel', 
#                                 sr=16000, n_fft=2048, 
#                                 fmin=0, fmax=8000, 
#                                 ax=ax2, cmap='magma')
# # p2 = librosa.display.specshow(batch_purified[0].squeeze().detach().cpu().numpy(), ax=ax2, y_axis='log', x_axis='time')
# plt.tight_layout()
# fig.colorbar(p1, ax=ax1, format="%+2.f dB")
# fig.colorbar(p2, ax=ax2, format="%+2.f dB")
# fig.savefig('spec.png')

# from torch.nn.modules.loss import _Loss
# class MarginalLoss(_Loss):

#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)

#     def forward(self, logits, targets):  # pylint: disable=arguments-differ
#         assert logits.shape[-1] >= 2
#         top_logits, top_classes = torch.topk(logits, 2, dim=-1)
#         target_logits = logits[torch.arange(logits.shape[0]), targets]
#         max_nontarget_logits = torch.where(
#             top_classes[..., 0] == targets,
#             top_logits[..., 1],
#             top_logits[..., 0],
#         )

#         loss = max_nontarget_logits - target_logits
#         if self.reduction == "none":
#             pass
#         elif self.reduction == "sum":
#             loss = loss.sum()
#         elif self.reduction == "mean":
#             loss = loss.mean()
#         else:
#             raise ValueError("unknown reduction: '%s'" % (self.recution,))

#         return loss