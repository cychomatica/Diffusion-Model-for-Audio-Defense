MODEL=resnext29_8_64
SIGMA=0.06

nohup \
~/anaconda3/envs/shutong/bin/python -u \
audio_models/ConvNets_SpeechCommands/adv_train_speech_commands.py \
--train_dataset=datasets/speech_commands/train \
--valid_dataset=datasets/speech_commands/valid \
--background_noise=datasets/speech_commands/train/_background_noise_ \
--model=$MODEL \
--gpu_id=0 \
--optim=sgd \
--lr_scheduler=plateau \
--learning_rate=0.01 \
--lr_scheduler_patience=5 \
--max_epochs=70 \
--batch_size=50 \
--sigma=$SIGMA \
> _Experiments/gaussian_sigma=$SIGMA\_train_sc09_$MODEL.log&