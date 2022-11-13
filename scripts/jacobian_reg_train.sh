set -e
MODEL=resnext29_8_64
LAMBDA=1e-11

nohup \
~/anaconda3/envs/shutong/bin/python -u \
audio_models/ConvNets_SpeechCommands/reg_train_speech_commands.py \
--train_dataset=datasets/speech_commands/train \
--valid_dataset=datasets/speech_commands/valid \
--background_noise=datasets/speech_commands/train/_background_noise_ \
--model=$MODEL \
--gpu_id=1 \
--optim=sgd \
--lr_scheduler=plateau \
--learning_rate=0.01 \
--lr_scheduler_patience=5 \
--max_epochs=70 \
--batch_size=96 \
--lambda_reg=$LAMBDA \
> _Experiments/jacobian\_reg=$LAMBDA\_train_sc09_$MODEL.log&