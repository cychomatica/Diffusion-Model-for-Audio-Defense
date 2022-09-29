set -e

MODEL_NAME=resnext29_8_64
MODEL_TYPE=vanilla

DIFFWAVE_PATH=diffusion_models/DiffWave_Unconditional/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl

DEFENSE=Diffusion
REVERSE_TIMESTEP=5
DEFENSE_NAME=$DEFENSE
if [ $DEFENSE == Diffusion -o Diffusion-Spec ]; then DEFENSE_NAME=$DEFENSE-t=$REVERSE_TIMESTEP; fi

ATTACK=FAKEBOB
START=90
END=99

GPU_ID=1
BATCH_SIZE=1

nohup \
~/anaconda3/envs/shutong/bin/python -u \
fakebob_eval.py \
--classifier_model $MODEL_NAME \
--classifier_type $MODEL_TYPE \
--defense $DEFENSE \
--ddpm_path $DIFFWAVE_PATH \
--idx_start $START \
--idx_end $END \
--t $REVERSE_TIMESTEP \
--attack $ATTACK \
--num_per_class 10 \
--gpu $GPU_ID \
--verbose 0 \
--batch_size $BATCH_SIZE \
> _Experiments/blackbox_attack/log/SC09-$START-$END-$MODEL_TYPE\_$MODEL_NAME-Defense=$DEFENSE_NAME-Attack=Adaptive_$ATTACK.log&