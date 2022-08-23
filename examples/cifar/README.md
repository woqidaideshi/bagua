python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm qadam > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main_yafen.py --algorithm qadam > /dev/null



python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm qgadam_low_precision_decentralized > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main_yafen.py --algorithm qgadam_low_precision_decentralized > /dev/null


python main_origin.py



sparse:
python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm gradient_allreduce --lr 0.05 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy --lr 0.1 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy-simple --lr 0.05 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm test --lr 0.05 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm test_simple --lr 0.05 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy-allgather --lr 0.05 > /dev/null

python -m bagua.distributed.launch --nproc_per_node=4 main.py --algorithm sparsepy-allgather-full --lr 0.05 > /dev/null



python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm gradient_allreduce --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm test --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-simple --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparsepy-allgather-full --lr 0.05 --epochs 50 > /dev/null


python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparse-test-inplace --lr 0.05 --epochs 50 > /dev/null
python -m bagua.distributed.launch --nproc_per_node=4 main_time.py --algorithm sparse-test --lr 0.05 --epochs 50 > /dev/null
