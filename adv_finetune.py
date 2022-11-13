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
    parser.add_argument("--train_path", default='datasets/speech_commands/train')
    parser.add_argument("--valid_path", default='datasets/speech_commands/valid')
    parser.add_argument("--test_path", default='datasets/speech_commands/test')
    parser.add_argument("--classifier_model", type=str, choices=['resnext29_8_64', 'vgg19_bn', 'densenet_bc_100_12', 'wideresnet28_10', 'm5'], default='resnext29_8_64')
    parser.add_argument("--classifier_type", type=str, choices=['advtr', 'vanilla'], default='vanilla')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--classifier_save_path", type=str, default='audio_models/ConvNets_SpeechCommands/checkpoints/adv_finetuned_model')
    parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--ddpm_path', type=str, default='diffusion_models/DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl')
    # parser.add_argument('--ddpm_path', type=str, default='diffusion_models/Improved_Diffusion_Unconditional/checkpoints/model084000.pt')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=10, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', action='store_true', default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=False, help='whether to use brownian motion')
    parser.add_argument("--ddpm_save_path", type=str, default='diffusion_models/DiffWave_Unconditional/exp/adv_finetuned_model')

    '''attack arguments'''
    parser.add_argument('--attack', type=str, choices=['CW', 'Qin-I', 'Kenansville', 'FAKEBOB', 'SirenAttack'], default='CW')
    parser.add_argument('--defense', type=str, choices=['Diffusion', 'Diffusion-Spec', 'AS', 'MS', 'DS', 'LPF', 'BPF', 'FeCo', 'None'], default='Diffusion')
    parser.add_argument('--bound_norm', type=str, choices=['linf', 'l2'], default='linf')
    parser.add_argument('--eps', type=int, default=131)
    parser.add_argument('--max_iter_1', type=int, default=10)
    parser.add_argument('--max_iter_2', type=int, default=0)
    parser.add_argument('--eot_attack_size', type=int, default=1)
    parser.add_argument('--eot_defense_size', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=2, help='batch size')
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

    Classifier = create_model(classifier_path)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()

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
    # AS_MODEL.eval()

    '''attack setting'''
    from robustness_eval.white_box_attack import *
    if args.attack == 'CW': # can be seen as PGD
        Attacker = AudioAttack(model=AS_MODEL, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=0,
                                learning_rate_1=args.eps/5 if args.bound_norm=='linf' else args.eps/50, 
                                eot_attack_size=args.eot_attack_size,
                                eot_defense_size=args.eot_defense_size,
                                verbose=args.verbose)
        print('attack: {} with {}_eps={} & iter={} & eot={}-{}\n'\
            .format(args.attack, args.bound_norm, args.eps, args.max_iter_1, args.eot_attack_size, args.eot_defense_size))
    # elif args.attack == 'Qin-I': # not used
    #     PsyMask = PsychoacousticMasker()
    #     Attacker = AudioAttack(model=AS_MODEL, masker=PsyMask, 
    #                             eps=args.eps, norm=args.bound_norm,
    #                             max_iter_1=args.max_iter_1, max_iter_2=args.max_iter_2,
    #                             learning_rate_1=args.eps/5,
    #                             verbose=args.verbose)
    # elif args.attack == 'Kenansville':
    #     method = 'ssa'
    #     max_iter = 30
    #     raster_width = 100
    #     Attacker = Kenansville(model=AS_MODEL, atk_name=method, 
    #                            max_iter=max_iter, raster_width=raster_width, 
    #                            verbose=args.verbose, batch_size=args.batch_size)
    #     print('attack: {} with method={} & raster_width={} & iter={}\n'\
    #         .format(args.attack, method, max_iter, raster_width))
    # elif args.attack == 'FAKEBOB':
    #     eps = 0.002 #args.eps / (2**15)
    #     confidence = 0.5
    #     max_iter = 200
    #     samples_per_draw = 200
    #     Attacker = FAKEBOB(model=AS_MODEL, task='SCR', targeted=False, verbose=args.verbose,
    #                        confidence=confidence, epsilon=eps, max_lr=5e-4, min_lr=1e-4,
    #                        max_iter=max_iter, samples_per_draw=samples_per_draw, samples_per_draw_batch_size=samples_per_draw, batch_size=args.batch_size)
    #     print('attack: {} with eps={} & confidence={} & iter={} & samples_per_draw={}\n'\
    #         .format(args.attack, eps, confidence, max_iter, samples_per_draw))
    # elif args.attack == 'SirenAttack':
    #     eps = 0.002 #args.eps / (2**15)
    #     max_epoch = 300
    #     max_iter = 30
    #     n_particles = 25
    #     Attacker = SirenAttack(model=AS_MODEL, task='SCR', targeted=False, verbose=args.verbose, batch_size=args.batch_size, 
    #                            epsilon=eps, max_epoch=max_epoch, max_iter=max_iter, n_particles=n_particles)
    #     print('attack: {} with eps={} & max_epoch={} & iter={} & n_particles={}\n'\
    #         .format(args.attack, eps, max_epoch, max_iter, n_particles))
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))


    
    transform = Compose([LoadAudio(), FixAudioLength()])

    train_dataset = SC09Dataset(folder=args.train_path, transform=transform, num_per_class=args.num_per_class)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    valid_dataset = SC09Dataset(folder=args.valid_path, transform=transform, num_per_class=args.num_per_class)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    test_dataset = SC09Dataset(folder=args.test_path, transform=transform, num_per_class=args.num_per_class)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

    '''adversarial finetune'''
    from tqdm import tqdm
    train_pbar = tqdm(train_dataloader, unit='audios', unit_scale=train_dataloader.batch_size)
    valid_pbar = tqdm(valid_dataloader, unit='audios', unit_scale=valid_dataloader.batch_size)

    AS_MODEL.train()
    finetune_epoch = 100
    criterion = torch.nn.CrossEntropyLoss()
    adv_finetune_opt = torch.optim.SGD(AS_MODEL.parameters(), lr=1e-3)

    for ep in range(finetune_epoch):

        '''finetuning'''
        for train_batch in train_pbar:

            waveforms = train_batch['samples']
            waveforms = torch.unsqueeze(waveforms, 1)
            targets = train_batch['target']

            waveforms = waveforms.cuda()
            targets = targets.cuda()

            waveforms_adv, _ = Attacker.generate(x=waveforms, y=targets, targeted=False)
            pred_adv = AS_MODEL(waveforms_adv)

            adv_finetune_loss = criterion(targets, pred_adv)
            adv_finetune_loss.backward()
            adv_finetune_opt.step()

        '''robustness validating'''
        for valid_batch in valid_pbar:

            waveforms = train_batch['samples']
            waveforms = torch.unsqueeze(waveforms, 1)
            targets = train_batch['target']

            waveforms = waveforms.cuda()
            targets = targets.cuda()

            waveforms_adv, attack_success = Attacker.generate(x=waveforms, y=targets, targeted=False)



    '''robustness eval'''
    test_pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_orig = 0
    correct_orig_denoised = 0
    correct_adv_1 = 0
    success_adv_2 = 0
    total = 0

    for test_batch in test_pbar:
        
        waveforms = test_batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = test_batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        '''original audio'''
        pred_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()
        # y_orig = torch.squeeze(Classifier(Wave2Spect(waveforms)).max(1, keepdim=True)[1])

        '''denoised original audio'''
        if AS_MODEL.defense_type == 'wave':
            if args.defense == 'None':
                waveforms_defended = waveforms
            else:
                waveforms_defended = AS_MODEL.defender(waveforms)
            pred_defended = AS_MODEL(waveforms_defended, False).max(1, keepdim=True)[1].squeeze()
        elif AS_MODEL.defense_type == 'spec': 
            spectrogram = AS_MODEL.transform(waveforms)
            if args.defense == 'None':
                spectrogram_defended = spectrogram
            else:
                spectrogram_defended = AS_MODEL.defender(spectrogram)
            pred_defended = AS_MODEL.classifier(spectrogram_defended).max(1, keepdim=True)[1].squeeze()
        # y_orig_denoised = torch.squeeze(Classifier(Wave2Spect(waveforms_defended)).max(1, keepdim=True)[1])

        '''adversarial audio'''
        waveforms_adv, attack_success = Attacker.generate(x=waveforms, y=targets, targeted=False)
        if isinstance(waveforms_adv, np.ndarray):
            if waveforms_adv.dtype == np.int16 and waveforms_adv.max() > 1 and waveforms_adv.min() < -1:
                waveforms_adv = waveforms_adv / (2**15)
            waveforms_adv = torch.tensor(waveforms_adv, dtype=waveforms.dtype).to(waveforms.device)
        
        '''for plot only'''
        waveforms_adv_diffusion = AS_MODEL.defender.model._diffusion(waveforms_adv)
        waveforms_adv_reverse = AS_MODEL.defender.model._reverse(waveforms_adv_diffusion)

        '''denoised adversarial audio'''
        if AS_MODEL.defense_type == 'wave':
            if args.defense == 'None':
                waveforms_adv_defended = waveforms_adv
            else:
                waveforms_adv_defended = AS_MODEL.defender(waveforms_adv)
        elif AS_MODEL.defense_type == 'spec':
            spectrogram_adv = AS_MODEL.transform(waveforms_adv)
            if args.defense == 'None':
                spectrogram_adv_defended = spectrogram_adv
            else:
                spectrogram_adv_defended = AS_MODEL.defender(spectrogram_adv)

        '''waveform/spectrogram saving'''
        if args.save_path is not None:
 
            clean_path = os.path.join(args.save_path,'clean')
            adv_path = os.path.join(args.save_path,'adv')

            if not os.path.exists(clean_path):
                os.makedirs(clean_path)
            if not os.path.exists(adv_path):
                os.makedirs(adv_path)

            for i in range(waveforms.shape[0]):
                
                audio_id = str(total + i).zfill(3)

                if AS_MODEL.defense_type == 'wave': 
                    utils.audio_save(waveforms[i], path=clean_path, 
                                     name='{}_{}_clean.wav'.format(audio_id,targets[i].item()))
                    utils.audio_save(waveforms_defended[i], path=clean_path, 
                                     name='{}_{}_clean_purified.wav'.format(audio_id,targets[i].item()))
                    utils.audio_save(waveforms_adv[i], path=adv_path, 
                                     name='{}_{}_adv.wav'.format(audio_id,targets[i].item()))
                    utils.audio_save(waveforms_adv[i], path=adv_path, 
                                     name='{}_{}_adv_purified.wav'.format(audio_id,targets[i].item()))
                elif AS_MODEL.defense_type == 'spec':
                    utils.spec_save(spectrogram[i], path=clean_path, 
                                    name='{}_{}_clean.png'.format(audio_id,targets[i].item()))
                    utils.spec_save(spectrogram_defended[i], path=clean_path, 
                                    name='{}_{}_clean_purified.png'.format(audio_id,targets[i].item()))
                    utils.spec_save(spectrogram_adv[i], path=adv_path, 
                                    name='{}_{}_adv.png'.format(audio_id,targets[i].item()))
                    utils.spec_save(spectrogram_adv_defended[i], path=adv_path, 
                                    name='{}_{}_adv_purified.png'.format(audio_id,targets[i].item()))



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

        test_pbar.set_postfix(pbar_info)
        test_pbar.update(1)


    '''summary'''
    print('on {} test examples: '.format(total))
    print('original clean test accuracy: {:.4f}%'.format(acc_orig))
    print('denoised clean test accuracy: {:.4f}%'.format(acc_orig_denoised))
    print('CW robust test accuracy: {:.4f}%'.format(acc_adv_1))
    if 'sr_adv_2' in dir():
        print('Imperceptible attack success rate: {:.4f}%'.format(sr_adv_2))

