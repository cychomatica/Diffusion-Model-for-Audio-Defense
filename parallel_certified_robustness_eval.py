import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

import numpy as np
import matplotlib.pyplot as plt

import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''SC09 classifier arguments'''
    parser.add_argument("--data_path", default='datasets/speech_commands/test')
    parser.add_argument("--victim_path", default='audio_models/ConvNets_SpeechCommands/checkpoints/sc09-resnext29_8_64_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02-best-acc.pth')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave arguments'''
    parser.add_argument('--config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--defender_path', type=str, default='diffusion_models/DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl')

    '''certified robust arguments'''
    parser.add_argument('--certified_example_id', type=int, default=7)
    parser.add_argument('--defense_method', type=str, default='diffusion', choices=['diffusion', 'randsmooth'])
    parser.add_argument('--sigma', type=float, default=0.25)
    parser.add_argument('--num_sampling', type=int, default=1000)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size')
    parser.add_argument('--gpu', type=int, default=2)

    '''file saving arguments'''
    parser.add_argument('--save_path', type=str, default='_Experiments/parallel_certified_robustness/records')

    args = parser.parse_args()


    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))


    '''SC09 classifier setting'''
    #region
    from transforms import *
    from datasets.sc_dataset import *
    from audio_models.ConvNets_SpeechCommands.create_model import *

    Classifier = create_model(args.victim_path)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()

    transform = Compose([LoadAudio(), FixAudioLength()])
    test_dataset = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=args.num_per_class)

    '''DiffWave denoiser setting'''
    #region
    from diffusion_models.diffwave_ddpm import create_diffwave_model
    DiffWave_Denoiser = create_diffwave_model(model_path=args.defender_path, config_path=args.config)
    DiffWave_Denoiser.eval().cuda()
    #endregion

    '''preprocessing setting'''
    #region
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])
    #endregion

    
    '''robust certificate setting'''
    from robustness_eval.certified_robust import *
    if args.defense_method == 'diffusion':
        RC = RobustCertificate(classifier=Classifier, transform=Wave2Spect, denoiser=DiffWave_Denoiser)
    elif args.defense_method == 'randsmooth':
        RC = RobustCertificate(classifier=Classifier, transform=Wave2Spect, denoiser=None)

    '''certified robustness eval'''
    example = test_dataset[args.certified_example_id]
    x = torch.tensor(example['samples'])[None,None,:]
    y = torch.tensor([example['target']])

    x = x.cuda()
    y = y.cuda()

    y_certified, r_certified = RC.certify(x=x, y=y, 
                                        sigma=args.sigma, n_0=100, 
                                        n=args.num_sampling, 
                                        batch_size=args.batch_size)

    '''save the record as json file'''
    record_save = {'id': args.certified_example_id, 
                   'y_true': y.item(), 
                   'y_pred': y_certified.item(), 
                   'certified_radius': r_certified.item()}
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    f = open(os.path.join(args.save_path, '{}.json'.format(args.certified_example_id)), 'w')
    json.dump(record_save, f)
    f.close()
    
