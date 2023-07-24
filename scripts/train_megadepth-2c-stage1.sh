SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"
cd $PROJECT_DIR

TRAIN_IMG_SIZE=704
TRAINING_STAGE=1
# to reproduced the results in our paper, please use:
data_cfg_path="configs/data/megadepth_trainval.py"
main_cfg_path="configs/model_configs/outdoor/loftr_ds_quadtree_cas_twins_large_stage4.py"

n_nodes=1
n_gpus_per_node=4
torch_num_workers=8
batch_size=2
pin_memory=true
exp_name="CasMTR-2c-stage${TRAINING_STAGE}-size${TRAIN_IMG_SIZE}-bs$(($n_gpus_per_node * $n_nodes * $batch_size))"

# stage1:8epoch, stage2,stage3:25epoch
python -u ./train.py \
  ${data_cfg_path} \
  ${main_cfg_path} \
  --exp_name=${exp_name} \
  --train_img_size=${TRAIN_IMG_SIZE} \
  --training_stage=${TRAINING_STAGE} \
  --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
  --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=100 \
  --flush_logs_every_n_steps=1 \
  --limit_val_batches=1. \
  --num_sanity_val_steps=2 \
  --benchmark=True \
  --max_epochs=8 \
  --precision=16 \
  --amp_level='O1'