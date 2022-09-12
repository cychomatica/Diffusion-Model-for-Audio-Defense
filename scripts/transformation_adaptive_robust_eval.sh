set -e

ATTACK=CW
BATCH_SIZE=8
REVERSE_TIMESTEP=25
GPU_ID=2
NORM=linf
EPS=65
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
--save_path _Experiments/transformation_defense/MS_adaptive_$ATTACK\_untargeted-$NORM-eps=$EPS-iter1=$MAX_ITER_1-iter2=$MAX_ITER_2 \
> _Experiments/transformation_defense/log/SC09-MS_Adaptive-$ATTACK-$NORM-eps=$EPS-iter1=$MAX_ITER_1-iter2=$MAX_ITER_2.log&