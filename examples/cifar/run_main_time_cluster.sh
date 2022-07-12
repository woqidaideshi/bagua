#!/bin/bash
# bash run_main_time_cluster.sh 4 0
nodes=$1
node_rank=$2

python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm gradient_allreduce --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm test --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy-simple --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy-allgather --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy-allgather-full --lr 0.05 --epochs 50 > /dev/null