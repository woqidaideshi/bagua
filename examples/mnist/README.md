Use the following script to start training locally with 8 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 main.py --algorithm gradient_allreduce
```

python -m bagua.distributed.launch --nproc_per_node=2 main.py --epochs 6 --algorithm low_precision_decentralized > /dev/null

python test.py --nprocs 2 --epochs 6 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm gradient_allreduce_sketch > /dev/null


python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm gradient_allreduce > /dev/null


python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm sketch > /dev/null