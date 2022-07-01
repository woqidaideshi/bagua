This script tests Bagua's low level communication primitives give the same result as PyTorch's. Use the following script to start testing locally with 4 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=4 main.py
```

python -m bagua.distributed.launch --nproc_per_node=4 test.py > test-100000-4.1.log

python -m bagua.distributed.launch --nproc_per_node=4 test.py > test_synchronize-100000-4.1.log

python -m bagua.distributed.launch --nproc_per_node=4 test.py > test_error-100000-4.2.log


python -m bagua.distributed.launch --nproc_per_node=4 main_other.py


python -m bagua.distributed.launch --nproc_per_node=4 test.py --func test_iwait
python -m bagua.distributed.launch --nproc_per_node=4 test.py > test_synchronize-100000-4.0608.log

python -m bagua.distributed.launch --nproc_per_node=4 test.py --func test_iwait > ./logs/test-test_iwait-0609.log

python test-bagua.py > ./logs/test-bagua-$(date +%m%d).1.log