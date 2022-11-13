set -e

nohup \
~/anaconda3/envs/shutong/bin/python -u \
distributed_train_qkws.py \
-c config_qkws.json \
> exp/ch256_T200_betaT0.02/logs/diffwave_unconditional_qkws.log&