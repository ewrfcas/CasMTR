#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR
export CUDA_VISIBLE_DEVICES=2

TEST_IMG_SIZE=640
data_cfg_path="configs/data/scannet_test_1500.py"
main_cfg_path="configs/model_configs/indoor/loftr_ds_quadtree_cas_stage3.py"
ckpt_path="pretrained_weights/CasMTR-indoor-4c/pl_checkpoint.ckpt"
dump_dir="outputs/CasMTR-indoor-4c-${TEST_IMG_SIZE}"
profiler_name="inference"
n_nodes=1 # mannually keep this the same with --nodes
n_gpus_per_node=1
torch_num_workers=4
batch_size=1 # per gpu

python -u ./test.py \
  ${data_cfg_path} \
  ${main_cfg_path} \
  --ckpt_path=${ckpt_path} \
  --dump_dir=${dump_dir} \
  --test_img_size=${TEST_IMG_SIZE} \
  --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
  --batch_size=${batch_size} --num_workers=${torch_num_workers} \
  --profiler_name=${profiler_name} \
  --benchmark \
  --refine
