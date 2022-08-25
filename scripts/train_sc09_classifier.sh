nohup \
~/anaconda3/envs/shutong/bin/python -u \
SpeechCommandsRecognition/train_speech_commands.py \
--train_dataset=/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/train \
--valid_dataset=/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/valid \
--background_noise=/home/shutong/project/Audio_Diffusion_Defense/Datasets/speech_commands/train/_background_noise_ \
--model=resnext29_8_64 \
--gpu_id=0 \
--optim=sgd \
--lr_scheduler=plateau \
--learning_rate=0.01 \
--lr_scheduler_patience=5 \
--max_epochs=70 \
--batch_size=96 \
> train_sc_classifier.log&