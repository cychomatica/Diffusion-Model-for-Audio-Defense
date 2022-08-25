set -e

ATTACK=CW
BATCH_SIZE=8
REVERSE_TIMESTEP=15
GPU_ID=0
EPS=2000
MAX_ITER_1=100

nohup \
~/anaconda3/envs/shutong/bin/python -u \
adaptive_attack_eval.py \
--attack $ATTACK \
--num_per_class 10 \
--batch_size $BATCH_SIZE \
--t $REVERSE_TIMESTEP \
--gpu $GPU_ID \
> log/SC09-VPSDE_Adaptive-$ATTACK-T=$REVERSE_TIMESTEP-eps=$EPS-iter=$MAX_ITER_1.log&