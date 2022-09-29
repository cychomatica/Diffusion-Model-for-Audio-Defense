# import os
# import argparse

# import torch
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision.transforms import *
# import torchaudio

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     '''SC09 classifier arguments'''
#     parser.add_argument("--data_path", default='Datasets/speech_commands/test')
#     parser.add_argument("--victim_path", default='checkpoints/model_stealing/1659540373400-modelstealing_resnext29_8_64_sgd_plateau_bs32_lr1.0e-02_wd1.0e-02-best-loss.pth')
#     parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')

#     '''surrogate arguments'''
#     parser.add_argument("--surrogate_path", default='checkpoints/sc-resnext29_8_64_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02-best-acc.pth')

#     '''DiffWave arguments'''
#     parser.add_argument('-c', '--config', type=str, default='config.json', help='JSON file for configuration')
#     parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
#     parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
#     parser.add_argument("--defender_path", default='DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl')
#     parser.add_argument('--reverse_timestep', type=int, default=10)

#     '''attack arguments'''
#     parser.add_argument('--attack', type=str, choices=['CW', 'SPSA'], default='CW')
#     parser.add_argument('--max_iter_1', type=int, default=1000)
#     parser.add_argument('--max_iter_2', type=int, default=0)

#     '''device arguments'''
#     parser.add_argument("--dataload_workers_nums", type=int, default=4, help='number of workers for dataloader')
#     parser.add_argument("--batch_size", type=int, default=64, help='batch size')
#     parser.add_argument('--gpu', type=int, default=0)

#     '''file saving arguments'''
#     parser.add_argument('--save_path', type=str, default='adv_examples/cw_model_stealing_untargeted')

#     args = parser.parse_args()


#     '''device setting'''
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#     use_gpu = torch.cuda.is_available()
#     print('use_gpu', use_gpu)
#     print('gpu id: {}'.format(args.gpu))


#     '''SC09 classifier setting'''
#     #region
#     from transforms import *
#     from datasets.sc_dataset import *

#     SC09_ResNeXt = torch.load(args.victim_path).module
#     SC09_ResNeXt.float()
#     SC09_ResNeXt.eval()

#     if use_gpu:
#         torch.backends.cudnn.benchmark = True
#         SC09_ResNeXt.cuda()

#     # feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
#     # transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
#     transform = Compose([LoadAudio(), FixAudioLength()])
#     test_dataset = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=100)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
#                                 pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
#     criterion = torch.nn.CrossEntropyLoss()
#     #endregion

#     '''SC surrogate setting'''
#     SC_Surrogate_ResNeXt = torch.load(args.surrogate_path).module
#     SC_Surrogate_ResNeXt.float()
#     SC_Surrogate_ResNeXt.eval()
#     if use_gpu:
#         SC_Surrogate_ResNeXt.cuda()


#     '''DiffWave denoiser setting'''
#     #region
#     import json
#     from diffwave_denoise import DiffDenoiser
#     from DiffWave_Unconditional.WaveNet import WaveNet_Speech_Commands as DiffWave
#     from DiffWave_Unconditional.util import calc_diffusion_hyperparams

#     with open(args.config) as f:
#             data = f.read()
#     config = json.loads(data)
#     global wavenet_config
#     wavenet_config          = config["wavenet_config"]      # to define wavenet
#     global diffusion_config
#     diffusion_config        = config["diffusion_config"]    # basic hyperparameters
#     global trainset_config
#     trainset_config         = config["trainset_config"]     # to load trainset
#     global diffusion_hyperparams 
#     diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config)

#     Defender_Net = DiffWave(**wavenet_config).cuda()
#     checkpoint = torch.load(args.defender_path)
#     Defender_Net.load_state_dict(checkpoint['model_state_dict'])
#     Denoiser = DiffDenoiser(model=Defender_Net, diffusion_hyperparams=diffusion_hyperparams, reverse_timestep=args.reverse_timestep)
#     #endregion

#     '''preprocessing setting'''
#     #region
#     n_mels = 32
#     if args.classifier_input == 'mel40':
#         n_mels = 40
#     MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
#     Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
#     Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])
#     #endregion

#     '''attack setting'''
#     from attack import *
#     if args.attack == 'CW':
#         Attacker = AudioAttack(model=SC_Surrogate_ResNeXt, transform=Wave2Spect, defender=None, max_iter_1=args.max_iter_1, max_iter_2=args.max_iter_2)
#     elif args.attack == 'SPSA':
#         Attacker = LinfSPSA(model=SC_Surrogate_ResNeXt, transform=Wave2Spect, defender=None)
#     else:
#         raise AttributeError("this version does not support '{}' at present".format(args.attack))

    
#     '''robustness eval'''
#     from tqdm import tqdm
#     pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

#     correct_orig = 0
#     correct_orig_denoised = 0
#     correct_adv = 0
#     correct_adv_denoised = 0
#     total = 0

#     for batch in pbar:
        
#         waveforms = batch['samples']
#         waveforms = torch.unsqueeze(waveforms, 1)
#         targets = batch['target']

#         waveforms = waveforms.cuda()
#         targets = targets.cuda()

#         '''original audio'''
#         y_orig = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms)).max(1, keepdim=True)[1])

#         '''denoised original audio'''
#         waveforms_denoised = Denoiser(waveforms=waveforms)
#         y_orig_denoised = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_denoised)).max(1, keepdim=True)[1])

#         '''adversarial audio'''
#         waveforms_adv, _ = Attacker.generate(x=waveforms, y=targets, targeted=False)
#         y_adv = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_adv)).max(1, keepdim=True)[1])

#         '''denoised adversarial audio'''
#         waveforms_adv_denoised = Denoiser(waveforms=waveforms_adv)
#         y_adv_denoised = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_adv_denoised)).max(1, keepdim=True)[1])

#         '''audio saving'''
#         if args.save_path is not None:
 
#             orig_path = os.path.join(args.save_path,'orig')
#             orig_denoised_path = os.path.join(args.save_path,'orig_denoised')
#             adv_path = os.path.join(args.save_path,'adv')
#             adv_denoised_path = os.path.join(args.save_path,'adv_denoised')

#             if not os.path.exists(orig_path):
#                 os.makedirs(orig_path)
#             if not os.path.exists(orig_denoised_path):
#                 os.makedirs(orig_denoised_path)
#             if not os.path.exists(adv_path):
#                 os.makedirs(adv_path)
#             if not os.path.exists(adv_denoised_path):
#                 os.makedirs(adv_denoised_path)

#             for i in range(waveforms.shape[0]):
                
#                 audio_id = str(total + i).zfill(3)

#                 torchaudio.save(os.path.join(orig_path,'{}_{}_orig.wav'.format(audio_id,targets[i].item())), 
#                                 waveforms[i].cpu(), batch['sample_rate'][i])
#                 torchaudio.save(os.path.join(orig_denoised_path,'{}_{}_orig_denoised.wav'.format(audio_id,targets[i].item())), 
#                                 waveforms_denoised[i].cpu(), batch['sample_rate'][i])
#                 torchaudio.save(os.path.join(adv_path,'{}_{}to{}_adv.wav'.format(audio_id,targets[i].item(),y_adv[i].item())), 
#                                 waveforms_adv[i].cpu(), batch['sample_rate'][i])
#                 torchaudio.save(os.path.join(adv_denoised_path,'{}_{}to{}_adv_denoised.wav'.format(audio_id,targets[i].item(),y_adv[i].item())), 
#                                 waveforms_adv_denoised[i].cpu(), batch['sample_rate'][i])



#         '''metrics output'''
#         correct_orig += (y_orig==targets).sum().item()
#         correct_orig_denoised += (y_orig_denoised==targets).sum().item()
#         correct_adv += (y_adv==targets).sum().item()
#         correct_adv_denoised += (y_adv_denoised==targets).sum().item()
#         total += waveforms.shape[0]

#         acc_orig = correct_orig / total * 100
#         acc_orig_denoised = correct_orig_denoised / total * 100
#         acc_adv = correct_adv / total * 100
#         acc_adv_denoised = correct_adv_denoised / total * 100
#         pbar.set_postfix(
#             {
#                 'orig acc: ': '{:.4f}%'.format(acc_orig),
#                 'denoised orig acc: ': '{:.4f}%'.format(acc_orig_denoised),
#                 'adv acc: ': '{:.4f}%'.format(acc_adv),
#                 'denoised adv acc: ': '{:.4f}%'.format(acc_adv_denoised)
#             }
#         )
#         pbar.update(1)

#     '''summary'''
#     print('on {} test examples: '.format(total))
#     print('original test accuracy: {:.4f}%'.format(acc_orig))
#     print('denoised original test accuracy: {:.4f}%'.format(acc_orig_denoised))
#     print('transfer adversarial test accuracy: {:.4f}%'.format(acc_adv))
#     print('denoised transfer adversarial test accuracy: {:.4f}%'.format(acc_adv_denoised))

import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

from diffusion_models.diffwave_ddpm import DiffWave
from robustness_eval.black_box_attack import FAKEBOB, Kenansville, SirenAttack

import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''SC09 classifier arguments'''
    parser.add_argument("--data_path", default='datasets/speech_commands/test')
    parser.add_argument("--classifier_model", type=str, choices=['resnext29_8_64', 'vgg19_bn', 'densenet_bc_100_12', 'wideresnet28_10', 'm5'], default='resnext29_8_64')
    parser.add_argument("--classifier_type", type=str, choices=['advtr', 'vanilla'], default='vanilla')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--ddpm_path', type=str, default='diffusion_models/DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl')
    # parser.add_argument('--ddpm_path', type=str, default='diffusion_models/Improved_Diffusion_Unconditional/checkpoints/model084000.pt')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=5, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', action='store_true', default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=False, help='whether to use brownian motion')

    '''attack arguments'''
    parser.add_argument('--attack', type=str, choices=['CW', 'Qin-I', 'Kenansville', 'FAKEBOB', 'SirenAttack'], default='CW')
    parser.add_argument('--defense', type=str, choices=['Diffusion', 'Diffusion-Spec', 'AS', 'MS', 'DS', 'LPF', 'BPF', 'FeCo', 'None'], default='Diffusion')
    parser.add_argument('--bound_norm', type=str, choices=['linf', 'l2'], default='linf')
    parser.add_argument('--eps', type=int, default=65)
    parser.add_argument('--max_iter_1', type=int, default=100)
    parser.add_argument('--max_iter_2', type=int, default=0)
    parser.add_argument('--eot_attack_size', type=int, default=1)
    parser.add_argument('--eot_defense_size', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=10, help='batch size')
    parser.add_argument('--gpu', type=int, default=1)

    '''file saving arguments'''
    parser.add_argument('--save_path', default='_Spec_Samples')

    args = parser.parse_args()


    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))


    '''
        set audio system model
    '''

    '''SC09 classifier setting'''
    from transforms import *
    from datasets.sc_dataset import *
    from audio_models.ConvNets_SpeechCommands.create_model import *

    if args.classifier_model == 'resnext29_8_64':
        classifier_path = 'audio_models/ConvNets_SpeechCommands/checkpoints/resnext29_8_64_sgd_plateau_bs64_lr1.0e-02_wd1.0e-02'
    elif args.classifier_model == 'vgg19_bn':
        classifier_path = 'audio_models/ConvNets_SpeechCommands/checkpoints/vgg19_bn_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02'
    elif args.classifier_model == 'densenet_bc_100_12':
        classifier_path = 'audio_models/ConvNets_SpeechCommands/checkpoints/densenet_bc_100_12_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02'
    elif args.classifier_model == 'wideresnet28_10':
        classifier_path = 'audio_models/ConvNets_SpeechCommands/checkpoints/wideresnet28_10_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02'
    elif args.classifier_model == 'm5':
        classifier_path = 'audio_models/M5/checkpoints/kernel_size=160'
    else:
        raise NotImplementedError(f'Unknown classifier model: {args.classifier_model}!')

    if args.classifier_type == 'vanilla': 
        classifier_path = os.path.join(classifier_path, 'vanilla-best-acc.pth')
    elif args.classifier_type == 'advtr': 
        classifier_path = os.path.join(classifier_path, 'advtr-best-acc.pth')
    else:
        raise NotImplementedError(f'Unknown classifier type: {args.classifier_type}!')

    # classifier_path = 'audio_models/ConvNets_SpeechCommands/checkpoints/resnext29_8_64_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02/1663209840164-best-acc.pth'
    Classifier = create_model(classifier_path)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()

    transform = Compose([LoadAudio(), FixAudioLength()])
    test_dataset = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=args.num_per_class)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    criterion = torch.nn.CrossEntropyLoss()

    '''preprocessing setting (if use acoustic features like mel-spectrogram)'''
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

    '''defense setting'''
    from acoustic_system import AcousticSystem
    if args.defense == 'None':
        if args.classifier_model == 'm5': # M5Net takes the raw audio as input
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=None, defender=None)
        else: 
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=Wave2Spect, defender=None)
        print('classifier model: {}'.format(Classifier._get_name()))
        print('classifier type: {}'.format(args.classifier_type))
        print('defense: None')

    else:
        if args.defense == 'Diffusion':
            from diffusion_models.diffwave_sde import *
            Defender = RevDiffWave(args)
            defense_type = 'wave'
        elif args.defense == 'Diffusion-Spec':
            from diffusion_models.improved_diffusion_sde import *
            Defender = RevImprovedDiffusion(args)
            defense_type = 'spec'
        elif args.defense == 'AS': 
            from transforms.time_defense import *
            Defender = TimeDomainDefense(defense_type='AS')
            defense_type = 'wave'
        elif args.defense == 'MS': 
            from transforms.time_defense import *
            Defender = TimeDomainDefense(defense_type='MS')
            defense_type = 'wave'
        elif args.defense == 'DS': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='DS')
            defense_type = 'wave'
        elif args.defense == 'LPF': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='LPF')
            defense_type = 'wave'
        elif args.defense == 'BPF': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='BPF')
            defense_type = 'wave'
        elif args.defense == 'FeCo':
            from transforms.feature_defense import *
            Defender = FeCo(param=0.2)
            defense_type = 'wave'
        else:
            raise NotImplementedError(f'Unknown defense: {args.defense}!')
        
        if args.classifier_model == 'm5':
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=None, defender=Defender, defense_type=defense_type)
        else: 
            AS_MODEL = AcousticSystem(classifier=Classifier, transform=Wave2Spect, defender=Defender, defense_type=defense_type)
        print('classifier model: {}'.format(Classifier._get_name()))
        print('classifier type: {}'.format(args.classifier_type))
        if args.defense == 'Diffusion':
            print('defense: {} with t={}'.format(Defender._get_name(), args.t))
        else:
            print('defense: {}'.format(Defender._get_name()))
    AS_MODEL.eval()


    if args.t == 1:
        surrogate_path = '_Experiments/model_stealing/model_stealing/T=1/best-loss-speech-commands-checkpoint-modelstealing_resnext29_8_64_sgd_plateau_bs32_lr1.0e-02_wd1.0e-02.pth'
        # surrogate_path = '_Experiments/model_stealing/model_stealing/T=1/1659540373400-modelstealing_resnext29_8_64_sgd_plateau_bs32_lr1.0e-02_wd1.0e-02-best-loss.pth'
    else:
        surrogate_path = '_Experiments/model_stealing/model_stealing/T=5/best-loss-speech-commands-checkpoint-modelstealing_resnext29_8_64_sgd_plateau_bs50_lr1.0e-02_wd1.0e-02.pth'
        # surrogate_path = '_Experiments/model_stealing/model_stealing/T=5/1659763954339-modelstealing_resnext29_8_64_sgd_plateau_bs50_lr1.0e-02_wd1.0e-02-best-loss.pth'
    from audio_models.ConvNets_SpeechCommands.models import CifarResNeXt
    Surrogate = CifarResNeXt(nlabels=10, in_channels=1)
    state_dict = dict(zip(Surrogate.state_dict().keys(), torch.load(surrogate_path)['state_dict'].values()))
    Surrogate.load_state_dict(state_dict)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Surrogate.cuda()
    Surr_MODEL = AcousticSystem(classifier=Surrogate, transform=Wave2Spect, defender=None)
    Surr_MODEL.eval()

    '''attack setting'''
    from robustness_eval.white_box_attack import *
    if args.attack == 'CW': # can be seen as PGD
        Attacker = AudioAttack(model=Surr_MODEL, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=0,
                                learning_rate_1=args.eps/5 if args.bound_norm=='linf' else args.eps/50, 
                                eot_attack_size=args.eot_attack_size,
                                eot_defense_size=args.eot_defense_size,
                                verbose=args.verbose)
        print('attack: {} with {}_eps={} & iter={} & eot={}-{}\n'\
            .format(args.attack, args.bound_norm, args.eps, args.max_iter_1, args.eot_attack_size, args.eot_defense_size))
    elif args.attack == 'Qin-I': # not used
        PsyMask = PsychoacousticMasker()
        Attacker = AudioAttack(model=Surr_MODEL, masker=PsyMask, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=args.max_iter_2,
                                learning_rate_1=args.eps/5,
                                verbose=args.verbose)
    elif args.attack == 'Kenansville':
        method = 'ssa'
        max_iter = 30
        raster_width = 100
        Attacker = Kenansville(model=Surr_MODEL, atk_name=method, 
                               max_iter=max_iter, raster_width=raster_width, 
                               verbose=args.verbose, batch_size=args.batch_size)
        print('attack: {} with method={} & raster_width={} & iter={}\n'\
            .format(args.attack, method, max_iter, raster_width))
    elif args.attack == 'FAKEBOB':
        eps = 0.002 #args.eps / (2**15)
        confidence = 0.5
        max_iter = 200
        samples_per_draw = 200
        Attacker = FAKEBOB(model=Surr_MODEL, task='SCR', targeted=False, verbose=args.verbose,
                           confidence=confidence, epsilon=eps, max_lr=5e-4, min_lr=1e-4,
                           max_iter=max_iter, samples_per_draw=samples_per_draw, samples_per_draw_batch_size=samples_per_draw, batch_size=args.batch_size)
        print('attack: {} with eps={} & confidence={} & iter={} & samples_per_draw={}\n'\
            .format(args.attack, eps, confidence, max_iter, samples_per_draw))
    elif args.attack == 'SirenAttack':
        eps = 0.002 #args.eps / (2**15)
        max_epoch = 300
        max_iter = 30
        n_particles = 25
        Attacker = SirenAttack(model=Surr_MODEL, task='SCR', targeted=False, verbose=args.verbose, batch_size=args.batch_size, 
                               epsilon=eps, max_epoch=max_epoch, max_iter=max_iter, n_particles=n_particles)
        print('attack: {} with eps={} & max_epoch={} & iter={} & n_particles={}\n'\
            .format(args.attack, eps, max_epoch, max_iter, n_particles))
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))


    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_steal = 0
    correct_vanilla_clean = 0
    correct_vanilla_robust = 0
    correct_defended_clean = 0
    correct_defended_robust = 0
    total = 0

    for batch in pbar:
        
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        '''original audio'''
        pred_vanilla_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()
        pred_defended_clean = AS_MODEL(waveforms, True).max(1, keepdim=True)[1].squeeze()
        pred_steal = Surr_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()

        '''adversarial audio'''
        waveforms_adv, _ = Attacker.generate(x=waveforms, y=targets, targeted=False)
        if isinstance(waveforms_adv, np.ndarray):
            if waveforms_adv.dtype == np.int16 and waveforms_adv.max() > 1 and waveforms_adv.min() < -1:
                waveforms_adv = waveforms_adv / (2**15)
            waveforms_adv = torch.tensor(waveforms_adv, dtype=waveforms.dtype).to(waveforms.device)
        
        pred_vanilla_adv = AS_MODEL(waveforms_adv, False).max(1, keepdim=True)[1].squeeze()
        pred_defended_adv = AS_MODEL(waveforms_adv, True).max(1, keepdim=True)[1].squeeze()


        '''metrics output'''
        total += waveforms.shape[0]
        correct_steal += (pred_steal==pred_defended_clean).sum().item()
        correct_vanilla_clean += (pred_vanilla_clean==targets).sum().item()
        correct_defended_clean += (pred_defended_clean==targets).sum().item()
        correct_vanilla_robust += (pred_vanilla_adv==targets).sum().item()
        correct_defended_robust += (pred_defended_adv==targets).sum().item()

        acc_steal = correct_steal / total * 100
        acc_vanilla_clean = correct_vanilla_clean / total * 100
        acc_defended_clean = correct_defended_clean / total * 100
        acc_vanilla_robust = correct_vanilla_robust / total * 100
        acc_defended_robust = correct_defended_robust / total * 100

        pbar_info = {
                    'acc_steal: ': '{:.4f}%'.format(acc_steal),
                    'acc_vanilla_clean: ': '{:.4f}%'.format(acc_vanilla_clean),
                    'acc_defended_clean: ': '{:.4f}%'.format(acc_defended_clean),
                    'acc_vanilla_robust: ': '{:.4f}%'.format(acc_vanilla_robust),
                    'acc_defended_robust: ': '{:.4f}%'.format(acc_defended_robust)
                    }

        pbar.set_postfix(pbar_info)
        pbar.update(1)


    '''summary'''
    print('on {} test examples: '.format(total))
    print('acc_steal: {:.4f}%'.format(acc_steal))
    print('acc_vanilla_clean: {:.4f}%'.format(acc_vanilla_clean))
    print('acc_defended_clean: {:.4f}%'.format(acc_defended_clean))
    print('acc_vanilla_robust: {:.4f}%'.format(acc_vanilla_robust))
    print('acc_defended_robust: {:.4f}%'.format(acc_defended_robust))
