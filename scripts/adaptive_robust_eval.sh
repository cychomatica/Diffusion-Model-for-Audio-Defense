set -e

MODEL_NAME=resnext29_8_64
MODEL_TYPE=vanilla

DEFENSE=Diffusion
REVERSE_TIMESTEP=5
DEFENSE_NAME=$DEFENSE
if [ $DEFENSE == Diffusion ]; then DEFENSE_NAME=$DEFENSE-t=$REVERSE_TIMESTEP; fi

ATTACK=CW
NORM=linf
EPS=65
MAX_ITER_1=100
MAX_ITER_2=0
EOT_ATTACK_SIZE=200
EOT_DEFENSE_SIZE=1

GPU_ID=3
BATCH_SIZE=10

nohup \
~/anaconda3/envs/shutong/bin/python -u \
adaptive_attack_eval.py \
--classifier_model $MODEL_NAME \
--classifier_type $MODEL_TYPE \
--defense $DEFENSE \
--t $REVERSE_TIMESTEP \
--attack $ATTACK \
--bound_norm $NORM \
--eps $EPS \
--max_iter_1 $MAX_ITER_1 \
--max_iter_2 $MAX_ITER_2 \
--eot_attack_size $EOT_ATTACK_SIZE \
--eot_defense_size $EOT_DEFENSE_SIZE \
--num_per_class 10 \
--gpu $GPU_ID \
--batch_size $BATCH_SIZE \
> _Experiments/eot_adaptive_attack/log/SC09-$MODEL_TYPE\_$MODEL_NAME-Defense=$DEFENSE_NAME-Attack=Adaptive_$ATTACK-EOT=$EOT_ATTACK_SIZE-$EOT_DEFENSE_SIZE-$NORM-eps=$EPS-iter1=$MAX_ITER_1-iter2=$MAX_ITER_2.log&