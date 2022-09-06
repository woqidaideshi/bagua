#!/bin/bash
# bash run_main_time_cluster.sh 4 0
set -e

nodes=$1
node_rank=$2

python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm gradient_allreduce --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm test --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy-simple --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy-allgather --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparsepy-allgather-full --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparse-test-inplace --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparse-test --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparse-py --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparse-py2 --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparse-py-independ --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparse-test-independ --epochs 20 > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --algorithm sparse-py-cuda --epochs 20 > /dev/null
