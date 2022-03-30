python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm qadam > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main_yafen.py --algorithm qadam > /dev/null



python -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm qgadam_low_precision_decentralized > /dev/null

python -m bagua.distributed.launch --nproc_per_node=2 main_yafen.py --algorithm qgadam_low_precision_decentralized > /dev/null


python main_origin.py
