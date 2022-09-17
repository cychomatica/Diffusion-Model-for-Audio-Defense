from ast import arg
import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

from diffusion_models.diffwave_ddpm import DiffWave
from robustness_eval.black_box_attack import FAKEBOB, Kenansville, SirenAttack

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
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=5, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', action='store_true', default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=False, help='whether to use brownian motion')

    '''attack arguments'''
    parser.add_argument('--attack', type=str, choices=['CW', 'Qin-I', 'Kenansville', 'FAKEBOB', 'SirenAttack'], default='CW')
    parser.add_argument('--defense', type=str, choices=['Diffusion', 'AS', 'MS', 'DS', 'LPF', 'BPF', 'FeCo', 'None'], default='None')
    parser.add_argument('--bound_norm', type=str, choices=['linf', 'l2'], default='linf')
    parser.add_argument('--eps', type=int, default=65)
    parser.add_argument('--max_iter_1', type=int, default=10)
    parser.add_argument('--max_iter_2', type=int, default=0)
    parser.add_argument('--eot_attack_size', type=int, default=1)
    parser.add_argument('--eot_defense_size', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=20, help='batch size')
    parser.add_argument('--gpu', type=int, default=1)

    '''file saving arguments'''
    parser.add_argument('--save_path', default=None)

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
    from audio_system import AudioSystem
    if args.defense == 'None':
        if args.classifier_model == 'm5': # M5Net takes the raw audio as input
            AS_MODEL = AudioSystem(classifier=Classifier, transform=None, defender=None)
        else: 
            AS_MODEL = AudioSystem(classifier=Classifier, transform=Wave2Spect, defender=None)
        print('classifier model: {}'.format(Classifier._get_name()))
        print('classifier type: {}'.format(args.classifier_type))
        print('defense: None')

    else:
        if args.defense == 'Diffusion':
            from diffusion_models.diffwave_sde import *
            Defender = RevDiffWave(args)
        elif args.defense == 'AS': 
            from transforms.time_defense import *
            Defender = TimeDomainDefense(defense_type='AS')
        elif args.defense == 'MS': 
            from transforms.time_defense import *
            Defender = TimeDomainDefense(defense_type='MS')
        elif args.defense == 'DS': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='DS')
        elif args.defense == 'LPF': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='LPF')
        elif args.defense == 'BPF': 
            from transforms.frequency_defense import *
            Defender = FreqDomainDefense(defense_type='BPF')
        elif args.defense == 'FeCo':
            from transforms.feature_defense import *
            Defender = FeCo(param=0.2)
        else:
            raise NotImplementedError(f'Unknown defense: {args.defense}!')
        
        if args.classifier_model == 'm5':
            AS_MODEL = AudioSystem(classifier=Classifier, transform=None, defender=Defender)
        else: 
            AS_MODEL = AudioSystem(classifier=Classifier, transform=Wave2Spect, defender=Defender)
        print('classifier model: {}'.format(Classifier._get_name()))
        print('classifier type: {}'.format(args.classifier_type))
        if args.defense == 'Diffusion':
            print('defense: {} with t={}'.format(Defender._get_name(), args.t))
        else:
            print('defense: {}'.format(Defender._get_name()))
    AS_MODEL.eval()

    '''attack setting'''
    from robustness_eval.white_box_attack import *
    if args.attack == 'CW': # can be seen as PGD
        Attacker = AudioAttack(model=AS_MODEL, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=0,
                                learning_rate_1=args.eps/5, 
                                eot_attack_size=args.eot_attack_size,
                                eot_defense_size=args.eot_defense_size,
                                verbose=args.verbose)
        print('attack: {} with {}_eps={} & iter={} & eot={}-{}\n'\
            .format(args.attack, args.bound_norm, args.eps, args.max_iter_1, args.eot_attack_size, args.eot_defense_size))
    elif args.attack == 'Qin-I': # not used
        PsyMask = PsychoacousticMasker()
        Attacker = AudioAttack(model=AS_MODEL, masker=PsyMask, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=args.max_iter_2,
                                learning_rate_1=args.eps/5,
                                verbose=args.verbose)
    elif args.attack == 'Kenansville':
        method = 'ssa'
        max_iter = 30
        raster_width = 100
        Attacker = Kenansville(model=AS_MODEL, atk_name=method, 
                               max_iter=max_iter, raster_width=raster_width, 
                               verbose=args.verbose, batch_size=args.batch_size)
        print('attack: {} with method={} & raster_width={} & iter={}\n'\
            .format(args.attack, method, max_iter, raster_width))
    elif args.attack == 'FAKEBOB':
        eps = 0.002 #args.eps / (2**15)
        confidence = 0.5
        max_iter = 200
        samples_per_draw = 50
        Attacker = FAKEBOB(model=AS_MODEL, task='SCR', targeted=False, verbose=args.verbose,
                           confidence=confidence, epsilon=eps, max_lr=5e-4, min_lr=1e-4,
                           max_iter=max_iter, samples_per_draw=samples_per_draw, batch_size=args.batch_size)
        print('attack: {} with eps={} & confidence={} & iter={} & samples_per_draw={}\n'\
            .format(args.attack, eps, confidence, max_iter, samples_per_draw))
    elif args.attack == 'SirenAttack':
        eps = 0.002 #args.eps / (2**15)
        max_epoch = 300
        max_iter = 30
        n_particles = 25
        Attacker = SirenAttack(model=AS_MODEL, task='SCR', targeted=False, verbose=args.verbose, batch_size=args.batch_size, 
                               epsilon=eps, max_epoch=max_epoch, max_iter=max_iter, n_particles=n_particles)
        print('attack: {} with eps={} & max_epoch={} & iter={} & n_particles={}\n'\
            .format(args.attack, eps, max_epoch, max_iter, n_particles))
    # elif args.attack == 'SPSA':
    #     Attacker = LinfSPSA(model=Classifier, transform=Wave2Spect, defender=DiffWave_VPSDE)
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))

    
    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_orig = 0
    correct_orig_denoised = 0
    correct_adv_1 = 0
    success_adv_2 = 0
    total = 0

    for batch in pbar:
        
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        '''original audio'''
        pred_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()
        # y_orig = torch.squeeze(Classifier(Wave2Spect(waveforms)).max(1, keepdim=True)[1])

        '''denoised original audio'''
        if args.defense == 'None':
            waveforms_defended = waveforms
        else:
            waveforms_defended = Defender(waveforms)
        pred_defended = AS_MODEL(waveforms_defended, False).max(1, keepdim=True)[1].squeeze()
        # y_orig_denoised = torch.squeeze(Classifier(Wave2Spect(waveforms_defended)).max(1, keepdim=True)[1])

        '''adversarial audio'''
        waveforms_adv, attack_success = Attacker.generate(x=waveforms, y=targets, targeted=False)
        if isinstance(waveforms_adv, np.ndarray):
            if waveforms_adv.dtype == np.int16:
                waveforms_adv = waveforms_adv / (2**15)
            waveforms_adv = torch.tensor(waveforms_adv, dtype=waveforms.dtype).to(waveforms.device)
            

        '''denoised adversarial audio'''
        if args.defense == 'None':
            waveforms_adv_defended = waveforms_adv
        else:
            waveforms_adv_defended = Defender(waveforms_adv)

        '''audio saving'''
        if args.save_path is not None:
 
            clean_path = os.path.join(args.save_path,'clean')
            clean_denoised_path = os.path.join(args.save_path,'clean_denoised')
            adv_path = os.path.join(args.save_path,'adv')
            adv_denoised_path = os.path.join(args.save_path,'adv_denoised')

            if not os.path.exists(clean_path):
                os.makedirs(clean_path)
            if not os.path.exists(clean_denoised_path):
                os.makedirs(clean_denoised_path)
            if not os.path.exists(adv_path):
                os.makedirs(adv_path)

            for i in range(waveforms.shape[0]):
                
                audio_id = str(total + i).zfill(3)

                torchaudio.save(os.path.join(clean_path,'{}_{}_clean.wav'.format(audio_id,targets[i].item())), 
                                waveforms[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(clean_denoised_path,'{}_{}_clean_denoised.wav'.format(audio_id,targets[i].item())), 
                                waveforms_defended[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(adv_path,'{}_{}_adv.wav'.format(audio_id,targets[i].item())), 
                                waveforms_adv[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(adv_path,'{}_{}_adv_denoised.wav'.format(audio_id,targets[i].item())), 
                                waveforms_adv_defended[i].cpu(), batch['sample_rate'][i])


        '''metrics output'''
        total += waveforms.shape[0]
        correct_orig += (pred_clean==targets).sum().item()
        correct_orig_denoised += (pred_defended==targets).sum().item()
        acc_orig = correct_orig / total * 100
        acc_orig_denoised = correct_orig_denoised / total * 100

        if isinstance(attack_success, tuple):
            correct_adv_1 += waveforms.shape[0] - torch.tensor(attack_success[0]).sum().item()
            acc_adv_1 = correct_adv_1 / total * 100
            pbar_info = {
                        'orig clean acc: ': '{:.4f}%'.format(acc_orig),
                        'denoised clean acc: ': '{:.4f}%'.format(acc_orig_denoised),
                        '{} robust acc: '.format(args.attack): '{:.4f}%'.format(acc_adv_1)
                        }
            if attack_success[1] is not None:
                success_adv_2 += torch.tensor(attack_success[1]).sum().item()
                sr_adv_2 = success_adv_2 / total * 100
                pbar_info.update({'ImpAtk success rate: ': '{:.4f}%'.format(sr_adv_2)})
        else:
            correct_adv_1 += waveforms.shape[0] - torch.tensor(attack_success).sum().item()
            acc_adv_1 = correct_adv_1 / total * 100

            pbar_info = {
                        'orig clean acc: ': '{:.4f}%'.format(acc_orig),
                        'denoised clean acc: ': '{:.4f}%'.format(acc_orig_denoised),
                        '{} robust acc: '.format(args.attack): '{:.4f}%'.format(acc_adv_1)
                        }

        pbar.set_postfix(pbar_info)
        pbar.update(1)


    '''summary'''
    print('on {} test examples: '.format(total))
    print('original clean test accuracy: {:.4f}%'.format(acc_orig))
    print('denoised clean test accuracy: {:.4f}%'.format(acc_orig_denoised))
    print('CW robust test accuracy: {:.4f}%'.format(acc_adv_1))
    if 'sr_adv_2' in dir():
        print('Imperceptible attack success rate: {:.4f}%'.format(sr_adv_2))

