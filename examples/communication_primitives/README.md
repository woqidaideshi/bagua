This script tests Bagua's low level communication primitives give the same result as PyTorch's. Use the following script to start testing locally with 4 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=4 main.py
```

python -m bagua.distributed.launch --nproc_per_node=4 test.py > test-100000-4.1.log
