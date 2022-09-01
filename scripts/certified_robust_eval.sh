set -e

BATCH_SIZE=50
<<<<<<< HEAD
GPU_ID=1
SIGMA=0.5
N=1000
=======
GPU_ID=0
SIGMA=0.5
N=100000
>>>>>>> 1e5d9fa75f7f49f08e326b6d703031023b802a2e
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
<<<<<<< HEAD
> _Experiments/certified_robustness/log/SC09-Certified-$DEFENSE-sigma=$SIGMA-N=$N.log&
=======
> _Experiments/certified_robustness/log/SC09-Certified-$DEFENSE-sigma=$SIGMA-N=$N.log&
>>>>>>> 1e5d9fa75f7f49f08e326b6d703031023b802a2e
