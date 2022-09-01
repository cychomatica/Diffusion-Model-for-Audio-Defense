set -e

ATTACK=CW
BATCH_SIZE=8
REVERSE_TIMESTEP=25
GPU_ID=0
NORM=linf
EPS=500
MAX_ITER_1=100
MAX_ITER_2=0

nohup \
~/anaconda3/envs/shutong/bin/python -u \
adaptive_attack_eval.py \
--attack $ATTACK \
--bound_norm $NORM \
--eps $EPS \
--max_iter_1 $MAX_ITER_1 \
--max_iter_2 $MAX_ITER_2 \
--num_per_class 10 \
--batch_size $BATCH_SIZE \
--t $REVERSE_TIMESTEP \
--gpu $GPU_ID \
--save_path _Experiments/adaptive_attack/vpsde-t=$REVERSE_TIMESTEP\_adaptive_$ATTACK\_untargeted-$NORM-eps=$EPS-iter1=$MAX_ITER_1-iter2=$MAX_ITER_2 \
> _Experiments/adaptive_attack/log/SC09-VPSDE_Adaptive-$ATTACK-T=$REVERSE_TIMESTEP-$NORM-eps=$EPS-iter1=$MAX_ITER_1-iter2=$MAX_ITER_2.log&