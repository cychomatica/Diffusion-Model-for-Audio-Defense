import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''SC09 classifier arguments'''
    parser.add_argument("--data_path", default='datasets/speech_commands/test')
    parser.add_argument("--victim_path", default='audio_models/ConvNets_SpeechCommands/checkpoints/sc09-resnext29_8_64_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02-best-acc.pth')
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
    parser.add_argument('--attack', type=str, choices=['CW', 'Qin-I', 'SPSA'], default='CW')
    parser.add_argument('--bound_norm', type=str, choices=['linf', 'l2'], default='l2')
    parser.add_argument('--eps', type=int, default=2000)
    parser.add_argument('--max_iter_1', type=int, default=10)
    parser.add_argument('--max_iter_2', type=int, default=10)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=8, help='batch size')
    parser.add_argument('--gpu', type=int, default=1)

    '''file saving arguments'''
    parser.add_argument('--save_path', default='_Experiments/adaptive_attack/vpsde_adaptive_Qin-I_untargeted')

    args = parser.parse_args()


    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))


    '''set audio system model'''
    # SC09 classifier setting
    from transforms import *
    from datasets.sc_dataset import *
    from audio_models.ConvNets_SpeechCommands.create_model import *
    Classifier = create_model(args.victim_path)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()
    # feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    # transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
    transform = Compose([LoadAudio(), FixAudioLength()])
    test_dataset = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=args.num_per_class)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    criterion = torch.nn.CrossEntropyLoss()

    # DiffWave denoiser setting
    from diffusion_models.diffwave_sde import RevGuidedDiffusion
    DiffWave_VPSDE = RevGuidedDiffusion(args=args)

    # preprocessing setting
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

    from audio_system import AudioSystem
    AS_MODEL = AudioSystem(classifier=Classifier, transform=Wave2Spect, defender=DiffWave_VPSDE)


    '''attack setting'''
    from robustness_eval.attack import *
    if args.attack == 'CW':
        Attacker = AudioAttack(model=AS_MODEL, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=0)
    elif args.attack == 'Qin-I':
        PsyMask = PsychoacousticMasker()
        Attacker = AudioAttack(model=AS_MODEL, masker=PsyMask, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=args.max_iter_2)
    elif args.attack == 'SPSA':
        Attacker = LinfSPSA(model=Classifier, transform=Wave2Spect, defender=DiffWave_VPSDE)
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))

    
    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_orig = 0
    correct_orig_denoised = 0
    correct_adv_1 = 0
    success_adv_2 = 0
    # correct_adv_denoised = 0
    total = 0

    for batch in pbar:
        
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        '''original audio'''
        y_orig = torch.squeeze(Classifier(Wave2Spect(waveforms)).max(1, keepdim=True)[1])

        '''denoised original audio'''
        waveforms_denoised = DiffWave_VPSDE(waveforms)
        y_orig_denoised = torch.squeeze(Classifier(Wave2Spect(waveforms_denoised)).max(1, keepdim=True)[1])

        '''adversarial audio'''
        # waveforms_adv, attack_success = Attacker.generate(x=waveforms, y=(targets+1)%10, targeted=True)
        waveforms_adv, attack_success_1, attack_success_2 = Attacker.generate(x=waveforms, y=targets, targeted=False)
        # y_adv = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_adv)).max(1, keepdim=True)[1])

        '''denoised adversarial audio'''
        waveforms_adv_denoised = DiffWave_VPSDE(waveforms_adv)
        # y_adv_denoised = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_adv_denoised)).max(1, keepdim=True)[1])

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
                                waveforms_denoised[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(adv_path,'{}_{}_adv.wav'.format(audio_id,targets[i].item())), 
                                waveforms_adv[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(adv_path,'{}_{}_adv_denoised.wav'.format(audio_id,targets[i].item())), 
                                waveforms_adv_denoised[i].cpu(), batch['sample_rate'][i])


        '''metrics output'''
        correct_orig += (y_orig==targets).sum().item()
        correct_orig_denoised += (y_orig_denoised==targets).sum().item()
        correct_adv_1 += waveforms.shape[0] - attack_success_1
        success_adv_2 += attack_success_2
        total += waveforms.shape[0]

        acc_orig = correct_orig / total * 100
        acc_orig_denoised = correct_orig_denoised / total * 100
        acc_adv_1 = correct_adv_1 / total * 100
        sr_adv_2 = success_adv_2 / total * 100
        pbar.set_postfix(
            {
                'orig clean acc: ': '{:.4f}%'.format(acc_orig),
                'denoised clean acc: ': '{:.4f}%'.format(acc_orig_denoised),
                'CW robust acc: ': '{:.4f}%'.format(acc_adv_1),
                'ImpAtk success rate: ': '{:.4f}%'.format(sr_adv_2),
            }
        )
        pbar.update(1)


    '''summary'''
    print('on {} test examples: '.format(total))
    print('original clean test accuracy: {:.4f}%'.format(acc_orig))
    print('denoised clean test accuracy: {:.4f}%'.format(acc_orig_denoised))
    print('CW robust test accuracy: {:.4f}%'.format(acc_adv_1))
    print('Imperceptible attack success rate:: {:.4f}%'.format(sr_adv_2))

