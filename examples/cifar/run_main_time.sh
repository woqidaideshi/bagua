#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,5

python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm gradient_allreduce --lr 0.05 --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm test --lr 0.05 --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-simple --lr 0.05 --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy --lr 0.05 --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather --lr 0.05 --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather-full --lr 0.05 --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparse-test-inplace --lr 0.05 --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparse-test --lr 0.05 --epochs 20 > /dev/null
