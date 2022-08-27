set -e

BATCH_SIZE=50
GPU_ID=0
SIGMA=0.5
N=100000
DEFENSE=diffusion

nohup \
~/anaconda3/envs/shutong/bin/python -u \
certified_robustness_eval.py \
--batch_size $BATCH_SIZE \
--defense_method $DEFENSE \
--sigma $SIGMA \
--num_sampling $N \
--gpu $GPU_ID \
--save_path _Experiments/certified_robustness/records \
> _Experiments/certified_robustness/log/SC09-Certified-$DEFENSE-sigma=$SIGMA-N=$N.log&
