set -e

BATCH_SIZE=50
REVERSE_TIMESTEP=5
GPU_ID=1

nohup \
~/anaconda3/envs/shutong/bin/python -u \
model_stealing.py \
--train_dataset=/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/train \
--valid_dataset=/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/valid \
--background_noise=/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/train/_background_noise_ \
--model=resnext29_8_64 \
--gpu_id=$GPU_ID \
--optim=sgd \
--lr_scheduler=plateau \
--learning_rate=0.01 \
--lr_scheduler_patience=5 \
--max_epochs=70 \
--batch_size=$BATCH_SIZE \
--reverse_timestep=$REVERSE_TIMESTEP \
--save_path=checkpoints/model_stealing/T=$REVERSE_TIMESTEP \
> log/model_stealing_T=$REVERSE_TIMESTEP.log&