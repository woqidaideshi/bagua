Use the following script to start training locally with 8 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 main.py --algorithm gradient_allreduce
```

python -m bagua.distributed.launch --nproc_per_node=2 main.py --epochs 6 --algorithm low_precision_decentralized > /dev/null

python test.py --nprocs 2 --epochs 6 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm gradient_allreduce_sketch > /dev/null

python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm gradient_allreduce > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm sketch > /dev/null

python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm floatgrad > /dev/null

python -m bagua.distributed.launch --nproc_per_node=3 main.py --algorithm qsparse > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 relaysum.py

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparse > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm signum > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm sketch-max > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main-max.py --algorithm sketch > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main_mod.py --algorithm qsparselocal > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 speed.py

python -m bagua.distributed.launch --nproc_per_node=2 main_marina.py --algorithm marina > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm marina --lr 0.05 > /dev/null


python -m bagua.distributed.launch --nproc_per_node=4 mnist_run.py --algorithm relay > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 cifar_run.py --algorithm relay > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 cifar_run.py --algorithm allreduce > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 cifar_run.py --algorithm marina --lr 0.05 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=1 test_fused_optimizer.py