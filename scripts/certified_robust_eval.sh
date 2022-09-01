set -e

BATCH_SIZE=50
GPU_ID=0
SIGMA=0.25
N=1000
DEFENSE=diffusion

nohup \
~/anaconda3/envs/shutong/bin/python -u \
certified_robustness_eval.py \
--batch_size $BATCH_SIZE \
--defense_method $DEFENSE \
--sigma $SIGMA \
--num_sampling $N \
--gpu $GPU_ID \
--save_path _Experiments/certified_robustness/records/2ShotsRev \
> _Experiments/certified_robustness/log/SC09-Certified-$DEFENSE-2ShotsRev-sigma=$SIGMA-N=$N.log&