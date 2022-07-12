#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,5

python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm gradient_allreduce --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm test --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-simple --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather-full --epochs 20 > /dev/null