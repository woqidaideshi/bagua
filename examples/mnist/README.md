Use the following script to start training locally with 8 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 main.py --algorithm gradient_allreduce
```

python -m bagua.distributed.launch --nproc_per_node=2 main.py --epochs 6 --algorithm low_precision_decentralized > /dev/null

python test.py --nprocs 2 --epochs 6 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm gradient_allreduce > /dev/null



test:
python -m bagua.distributed.launch --nproc_per_node=1 test_fused_optimizer.py

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm test

python3 -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm gradient_allreduce --lr 0.05 > /dev/null

python3 -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm gradient_allreduce_sgd --lr 0.05 > /dev/null

python3 -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm test_simple --lr 0.05 > /dev/null



sparse:
python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy-simple > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy-allgather > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy-allgather-full > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather-full > /dev/null


python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm gradient_allreduce --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm test --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-simple --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather --epochs 20 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather-full  --epochs 20 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm sparse-test-inplace > log/sparsepy/sparse-test-20220823.log

python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm sparse-test > log/sparsepy/sparse-test-20220823.log

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparse-test > log/sparsepy/sparse-test-20220823.log

python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm sparse-py > log/sparsepy/sparse-test-20220830.log

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparse-py2 > log/sparsepy/sparse-test-20220901.log

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparse-py-independ > log/sparsepy/sparse-test-20220901.log

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparse-test-independ > log/sparsepy/sparse-test-20220901.log

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparse-py-cuda > log/sparsepy/sparse-test-20220906.log

sketch:
python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm gradient_allreduce_sketch > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm sketch > /dev/null

python -m bagua.distributed.launch --nproc_per_node=1 main.py --algorithm floatgrad > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm sketch-max > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main-max.py --algorithm sketch > /dev/null



signum:
python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm signum > /dev/null



qsparse:
python -m bagua.distributed.launch --nproc_per_node=2 main_mod.py --algorithm qsparselocal > /dev/null

python -m bagua.distributed.launch --nproc_per_node=3 main.py --algorithm qsparse > /dev/null

python3 -m bagua.distributed.launch --nproc_per_node=1 benchmark.py --epochs 4 --lr 0.01 --algorithm gradient_allreduce

python3 -m bagua.distributed.launch --nproc_per_node=4 main_mod_vgg16.py --epochs 1 --lr 0.2 --algorithm gradient_allreduce

python3 -m bagua.distributed.launch --nproc_per_node=4 main_mod_vgg16.py --epochs 1 --lr 0.01 --algorithm qsparselocal  --gap 7 > /dev/null


relaysum:
python -m bagua.distributed.launch --nproc_per_node=4 relaysum.py

python -m bagua.distributed.launch --nproc_per_node=4 speed.py

cd ../../../RelaySGD
python -m bagua.distributed.launch --nproc_per_node=4 mnist_run.py --algorithm relay > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 cifar_run.py --algorithm relay > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 cifar_run.py --algorithm allreduce > /dev/null



marina:
python -m bagua.distributed.launch --nproc_per_node=2 main_marina.py --algorithm marina > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm marina --lr 0.05 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm marina --lr 0.05 > /dev/null

cd cifar
python -m bagua.distributed.launch --nproc_per_node=4 cifar_run.py --algorithm marina --lr 0.05 > /dev/null

cd experiments-marina/cifar/
python -m bagua.distributed.launch --nproc_per_node=4 cifar10_run.py --algorithm marina --lr 0.05 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 mnist_run.py --algorithm marina --lr 0.1 > /dev/null

cd ../../../RelaySGD
python -m bagua.distributed.launch --nproc_per_node=4 cifar_run.py --algorithm marina --lr 0.05 > /dev/null


