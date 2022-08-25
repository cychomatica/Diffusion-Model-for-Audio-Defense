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
    parser.add_argument("--data_path", default='Datasets/speech_commands/test')
    parser.add_argument("--victim_path", default='checkpoints/model_stealing/1659540373400-modelstealing_resnext29_8_64_sgd_plateau_bs32_lr1.0e-02_wd1.0e-02-best-loss.pth')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')

    '''surrogate arguments'''
    parser.add_argument("--surrogate_path", default='checkpoints/sc-resnext29_8_64_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02-best-acc.pth')

    '''DiffWave arguments'''
    parser.add_argument('-c', '--config', type=str, default='config.json', help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument("--defender_path", default='DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl')
    parser.add_argument('--reverse_timestep', type=int, default=10)

    '''attack arguments'''
    parser.add_argument('--attack', type=str, choices=['CW', 'SPSA'], default='CW')
    parser.add_argument('--max_iter_1', type=int, default=1000)
    parser.add_argument('--max_iter_2', type=int, default=0)

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    parser.add_argument('--gpu', type=int, default=0)

    '''file saving arguments'''
    parser.add_argument('--save_path', type=str, default='adv_examples/cw_model_stealing_untargeted')

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

    SC09_ResNeXt = torch.load(args.victim_path).module
    SC09_ResNeXt.float()
    SC09_ResNeXt.eval()

    if use_gpu:
        torch.backends.cudnn.benchmark = True
        SC09_ResNeXt.cuda()

    # feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    # transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
    transform = Compose([LoadAudio(), FixAudioLength()])
    test_dataset = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=100)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    criterion = torch.nn.CrossEntropyLoss()
    #endregion

    '''SC surrogate setting'''
    SC_Surrogate_ResNeXt = torch.load(args.surrogate_path).module
    SC_Surrogate_ResNeXt.float()
    SC_Surrogate_ResNeXt.eval()
    if use_gpu:
        SC_Surrogate_ResNeXt.cuda()


    '''DiffWave denoiser setting'''
    #region
    import json
    from diffwave_denoise import DiffDenoiser
    from DiffWave_Unconditional.WaveNet import WaveNet_Speech_Commands as DiffWave
    from DiffWave_Unconditional.util import calc_diffusion_hyperparams

    with open(args.config) as f:
            data = f.read()
    config = json.loads(data)
    global wavenet_config
    wavenet_config          = config["wavenet_config"]      # to define wavenet
    global diffusion_config
    diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    global trainset_config
    trainset_config         = config["trainset_config"]     # to load trainset
    global diffusion_hyperparams 
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config)

    Defender_Net = DiffWave(**wavenet_config).cuda()
    checkpoint = torch.load(args.defender_path)
    Defender_Net.load_state_dict(checkpoint['model_state_dict'])
    Denoiser = DiffDenoiser(model=Defender_Net, diffusion_hyperparams=diffusion_hyperparams, reverse_timestep=args.reverse_timestep)
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

    '''attack setting'''
    from attack import *
    if args.attack == 'CW':
        Attacker = AudioAttack(model=SC_Surrogate_ResNeXt, transform=Wave2Spect, defender=None, max_iter_1=args.max_iter_1, max_iter_2=args.max_iter_2)
    elif args.attack == 'SPSA':
        Attacker = LinfSPSA(model=SC_Surrogate_ResNeXt, transform=Wave2Spect, defender=None)
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))

    
    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_orig = 0
    correct_orig_denoised = 0
    correct_adv = 0
    correct_adv_denoised = 0
    total = 0

    for batch in pbar:
        
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        '''original audio'''
        y_orig = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms)).max(1, keepdim=True)[1])

        '''denoised original audio'''
        waveforms_denoised = Denoiser(waveforms=waveforms)
        y_orig_denoised = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_denoised)).max(1, keepdim=True)[1])

        '''adversarial audio'''
        waveforms_adv, _ = Attacker.generate(x=waveforms, y=targets, targeted=False)
        y_adv = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_adv)).max(1, keepdim=True)[1])

        '''denoised adversarial audio'''
        waveforms_adv_denoised = Denoiser(waveforms=waveforms_adv)
        y_adv_denoised = torch.squeeze(SC09_ResNeXt(Wave2Spect(waveforms_adv_denoised)).max(1, keepdim=True)[1])

        '''audio saving'''
        if args.save_path is not None:
 
            orig_path = os.path.join(args.save_path,'orig')
            orig_denoised_path = os.path.join(args.save_path,'orig_denoised')
            adv_path = os.path.join(args.save_path,'adv')
            adv_denoised_path = os.path.join(args.save_path,'adv_denoised')

            if not os.path.exists(orig_path):
                os.makedirs(orig_path)
            if not os.path.exists(orig_denoised_path):
                os.makedirs(orig_denoised_path)
            if not os.path.exists(adv_path):
                os.makedirs(adv_path)
            if not os.path.exists(adv_denoised_path):
                os.makedirs(adv_denoised_path)

            for i in range(waveforms.shape[0]):
                
                audio_id = str(total + i).zfill(3)

                torchaudio.save(os.path.join(orig_path,'{}_{}_orig.wav'.format(audio_id,targets[i].item())), 
                                waveforms[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(orig_denoised_path,'{}_{}_orig_denoised.wav'.format(audio_id,targets[i].item())), 
                                waveforms_denoised[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(adv_path,'{}_{}to{}_adv.wav'.format(audio_id,targets[i].item(),y_adv[i].item())), 
                                waveforms_adv[i].cpu(), batch['sample_rate'][i])
                torchaudio.save(os.path.join(adv_denoised_path,'{}_{}to{}_adv_denoised.wav'.format(audio_id,targets[i].item(),y_adv[i].item())), 
                                waveforms_adv_denoised[i].cpu(), batch['sample_rate'][i])



        '''metrics output'''
        correct_orig += (y_orig==targets).sum().item()
        correct_orig_denoised += (y_orig_denoised==targets).sum().item()
        correct_adv += (y_adv==targets).sum().item()
        correct_adv_denoised += (y_adv_denoised==targets).sum().item()
        total += waveforms.shape[0]

        acc_orig = correct_orig / total * 100
        acc_orig_denoised = correct_orig_denoised / total * 100
        acc_adv = correct_adv / total * 100
        acc_adv_denoised = correct_adv_denoised / total * 100
        pbar.set_postfix(
            {
                'orig acc: ': '{:.4f}%'.format(acc_orig),
                'denoised orig acc: ': '{:.4f}%'.format(acc_orig_denoised),
                'adv acc: ': '{:.4f}%'.format(acc_adv),
                'denoised adv acc: ': '{:.4f}%'.format(acc_adv_denoised)
            }
        )
        pbar.update(1)

    '''summary'''
    print('on {} test examples: '.format(total))
    print('original test accuracy: {:.4f}%'.format(acc_orig))
    print('denoised original test accuracy: {:.4f}%'.format(acc_orig_denoised))
    print('transfer adversarial test accuracy: {:.4f}%'.format(acc_adv))
    print('denoised transfer adversarial test accuracy: {:.4f}%'.format(acc_adv_denoised))

