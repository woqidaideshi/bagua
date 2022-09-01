#!/bin/bash
# bash run_main_time.sh ./imagenette2
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,5
datadir=$1

python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm gradient_allreduce --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm test --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy-simple --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy-allgather --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparsepy-allgather-full --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparse-test-inplace --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparse-test --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparse-py --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparse-py2 --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparse-py-independ --epochs 50 $datadir > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --arch resnet50 --algorithm sparse-test-independ --epochs 50 $datadir > /dev/null
