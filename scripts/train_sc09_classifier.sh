MODEL=resnext29_8_64
# SIGMA=0.0

nohup \
~/anaconda3/envs/shutong/bin/python -u \
audio_models/ConvNets_SpeechCommands/adv_train_speech_commands.py \
--train_dataset=datasets/speech_commands/train \
--valid_dataset=datasets/speech_commands/valid \
--background_noise=datasets/speech_commands/train/_background_noise_ \
--model=$MODEL \
--gpu_id=2 \
--optim=sgd \
--lr_scheduler=plateau \
--learning_rate=0.01 \
--lr_scheduler_patience=5 \
--max_epochs=70 \
--batch_size=96 \
> _Experiments/advtr\_train_sc09_$MODEL.log&

# MODEL=resnext29_8_64

# nohup \
# ~/anaconda3/envs/shutong/bin/python -u \
# audio_models/ConvNets_SpeechCommands/train_speech_commands.py \
# --train_dataset=datasets/speech_commands/train \
# --valid_dataset=datasets/speech_commands/valid \
# --background_noise=datasets/speech_commands/train/_background_noise_ \
# --model=$MODEL \
# --gpu_id=1 \
# --optim=sgd \
# --lr_scheduler=plateau \
# --learning_rate=0.01 \
# --lr_scheduler_patience=5 \
# --max_epochs=70 \
# --batch_size=96 \
# > _Experiments/train_sc09_$MODEL.log&