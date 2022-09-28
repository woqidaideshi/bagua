#!/bin/bash
# bash run_main_time_cluster.sh 4 0 ./imagenette2
set -e
nodes=$1
node_rank=$2
datadir=$3

python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm gradient_allreduce --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm test --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparsepy-simple --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparsepy --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparsepy-allgather --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparsepy-allgather-full --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-test-inplace --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-test --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-py --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-py2 --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-py-independ --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-test-independ --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-py-cuda --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-py-cuda-parallel --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-test-inplace-parallel --epochs 50 $datadir > /dev/null
sleep 5
python -m bagua.distributed.launch --nproc_per_node=1 --nnodes=$nodes --node_rank=$node_rank --master_addr="172.31.21.207" --master_port=1234 main_time.py --arch resnet50 --algorithm sparse-py-cuda-independ-parallel --epochs 50 $datadir > /dev/null
