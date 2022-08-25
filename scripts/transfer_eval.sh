set -e

ATTACK=CW
MAX_ITER_1=1000
BATCH_SIZE=32
REVERSE_TIMESTEP=5
GPU_ID=1

nohup \
~/anaconda3/envs/shutong/bin/python -u \
sc_transfer_eval.py \
--surrogate_path checkpoints/model_stealing/T=5/1659763954339-modelstealing_resnext29_8_64_sgd_plateau_bs50_lr1.0e-02_wd1.0e-02-best-loss.pth \
--attack $ATTACK \
--max_iter_1 $MAX_ITER_1 \
--batch_size $BATCH_SIZE \
--reverse_timestep $REVERSE_TIMESTEP \
--gpu $GPU_ID \
--save_path adv_examples/cw_model_stealing_untargeted \
> log/SC09-ModelStealingTransfer-$ATTACK-T=$REVERSE_TIMESTEP-iter=$MAX_ITER_1.log&