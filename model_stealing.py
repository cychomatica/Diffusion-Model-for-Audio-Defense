#!/usr/bin/env python
"""Train a CNN for Google speech commands."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import argparse
import time

from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision, torchaudio
from torchvision.transforms import *

# from tensorboardX import SummaryWriter

import audio_models.ConvNets_SpeechCommands.models as models
# from github_repo.SCRec_github.datasets import *
# from github_repo.SCRec_github.transforms import *
from audio_models.ConvNets_SpeechCommands.mixup import *
from datasets.sc_dataset import SC09_CLASSES

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

'''training setting'''
parser.add_argument("--train_dataset", type=str, default='/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/train', help='path of train dataset')
parser.add_argument("--valid_dataset", type=str, default='/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/valid', help='path of validation dataset')
parser.add_argument("--background_noise", type=str, default='/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch_size", type=int, default=32, help='batch size')
parser.add_argument("--dataload_workers_nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--weight_decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='learning rate for optimization')
parser.add_argument("--lr_scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr_scheduler_patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr_scheduler_step_size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr_scheduler_gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max_epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[9], help='model of NN')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
parser.add_argument('--mixup', action='store_true', help='use mixup')
parser.add_argument('--gpu_id', type=int, default=1)

'''SC09 classifier arguments'''
parser.add_argument("--victim_path", default='checkpoints/sc09-resnext29_8_64_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02-best-acc.pth')
parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')

'''DiffWave arguments'''
parser.add_argument('-c', '--config', type=str, default='config.json', help='JSON file for configuration')
parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
parser.add_argument("--defender_path", default='DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl')
parser.add_argument('--reverse_timestep', type=int, default=1)

parser.add_argument("--save_path", type=str, default='checkpoints')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True

'''dataset setting'''
#region
n_mels = 32
if args.input == 'mel40':
    n_mels = 40
MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

from transforms import *
from datasets.sc_dataset import *

transform = Compose([LoadAudio(), FixAudioLength()])
train_dataset = SC09Dataset(folder=args.train_dataset, transform=transform, num_per_class=74751)
valid_dataset = SC09Dataset(folder=args.valid_dataset, transform=transform, num_per_class=74751) 

weights = train_dataset.make_weights_for_balanced_classes()
sampler = WeightedRandomSampler(weights, len(weights))
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

#endregion


'''victim model and defender'''
#region
SC09_ResNeXt = torch.load(args.victim_path).module
SC09_ResNeXt.float()
SC09_ResNeXt.eval()
if use_gpu: 
    SC09_ResNeXt.cuda()


from diffusion_models.diffwave_ddpm import create_diffwave_model
DiffWave_Denoiser = create_diffwave_model(model_path=args.defender_path, config_path=args.config, reverse_timestep=args.reverse_timestep)
#endregion

# a name used to save checkpoints etc.
full_name = 'modelstealing_%s_%s_%s_bs%d_lr%.1e_wd%.1e' % (args.model, args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
if args.comment:
    full_name = '%s_%s' % (full_name, args.comment)

model = models.create_model(model_name=args.model, num_classes=len(SC09_CLASSES), in_channels=1)

if use_gpu:
    model = torch.nn.DataParallel(model).cuda()

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

start_timestamp = int(time.time()*1000)
start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0

if args.resume:
    print("resuming a checkpoint '%s'" % args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_accuracy = checkpoint.get('accuracy', best_accuracy)
    best_loss = checkpoint.get('loss', best_loss)
    start_epoch = checkpoint.get('epoch', start_epoch)
    global_step = checkpoint.get('step', global_step)

    del checkpoint  # reduce memory

if args.lr_scheduler == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

def get_lr():
    return optimizer.param_groups[0]['lr']

# writer = SummaryWriter(comment=('_speech_commands_' + full_name))

def train(epoch):
    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    # writer.add_scalar('%s/learning_rate' % phase,  get_lr(), epoch)

    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)

    for batch in pbar:
        inputs = batch['samples']
        inputs = torch.unsqueeze(inputs, 1)
        if use_gpu:
            inputs = inputs.cuda()

        targets = SC09_ResNeXt(Wave2Spect(DiffWave_Denoiser(inputs)))

        if args.mixup:
            inputs, targets = mixup(inputs, targets, num_classes=len(CLASSES))

        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)

    
        # forward/backward
        outputs = model(Wave2Spect(inputs))
        if args.mixup:
            loss = mixup_cross_entropy_loss(outputs, targets)
        else:
            loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        if args.mixup:
            targets = batch['target']
            targets = Variable(targets, requires_grad=False).cuda(non_blocking=True)
        correct += pred.eq(targets.data.max(1, keepdim=True)[1].view_as(pred)).sum().item()
        total += targets.size(0)


        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it

    print('\nepoch {}'.format(epoch))
    print('train_accuracy: {}\t train_loss: {}\n'.format(accuracy, epoch_loss))
    # writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    # writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

def valid(epoch):
    global best_accuracy, best_loss, global_step

    phase = 'valid'
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)

    for batch in pbar:

        inputs = batch['samples']
        inputs = torch.unsqueeze(inputs, 1)
        if use_gpu:
            inputs = inputs.cuda()
        # targets = batch['target']
        targets = SC09_ResNeXt(Wave2Spect(DiffWave_Denoiser(inputs)))

        inputs = Variable(inputs, requires_grad=False)
        targets = Variable(targets, requires_grad=False)


        # forward
        outputs = model(Wave2Spect(inputs))
        loss = criterion(outputs, targets)

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.max(1, keepdim=True)[1].view_as(pred)).sum().item()
        total += targets.size(0)

        # writer.add_scalar('%s/loss' % phase, loss.data[0], global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it

    print('\nepoch {}'.format(epoch))
    print('val_accuracy: {}\t val_loss: {}\n'.format(accuracy, epoch_loss))
    # writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    # writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer' : optimizer.state_dict(),
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, os.path.join(args.save_path, 'best-loss-speech-commands-checkpoint-%s.pth' % full_name))
        torch.save(model, os.path.join(args.save_path, '%d-%s-best-loss.pth' % (start_timestamp, full_name)))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, os.path.join(args.save_path, 'best-acc-speech-commands-checkpoint-%s.pth' % full_name))
        torch.save(model, os.path.join(args.save_path, '%d-%s-best-acc.pth' % (start_timestamp, full_name)))

    torch.save(checkpoint, os.path.join(args.save_path, 'last-speech-commands-checkpoint.pth'))
    del checkpoint  # reduce memory

    return epoch_loss

print("training %s for Google speech commands..." % args.model)
since = time.time()
for epoch in range(start_epoch, args.max_epochs):
    if args.lr_scheduler == 'step':
        lr_scheduler.step()

    train(epoch)
    epoch_loss = valid(epoch)

    if args.lr_scheduler == 'plateau':
        lr_scheduler.step(metrics=epoch_loss)

    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
    print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
print("finished")
