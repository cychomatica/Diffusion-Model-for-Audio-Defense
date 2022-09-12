set -e

BATCH_SIZE=50
GPU_ID=1
SIGMA=0.06
N=100000
DEFENSE=randsmooth

nohup \
~/anaconda3/envs/shutong/bin/python -u \
certified_robustness_eval.py \
--batch_size $BATCH_SIZE \
--defense_method $DEFENSE \
--sigma $SIGMA \
--num_sampling $N \
--gpu $GPU_ID \
--save_path _Experiments/certified_robustness/records/randsmooth_gaussian_aug_N=$N \
> _Experiments/certified_robustness/log/SC09-Certified-GaussianAug-$DEFENSE-sigma=$SIGMA-N=$N.log&

