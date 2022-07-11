#!/bin/bash
# bash run_main_time.sh ./imagenette2

export CUDA_VISIBLE_DEVICES=0,1,2,5
datadir=$1

python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm gradient_allreduce --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm test --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy-simple --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy-allgather --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy-allgather-full --epochs 50 $datadir > /dev/null