#!/bin/bash

set -e

# funcs=("test" "test_iwait" "test_iwaitlist" "test_synchronize" "test_synchronize_iwait" "test_synchronize_iwaitlist")
funcs=("test_iwait" "test_synchronize" "test_synchronize_iwait" "test_synchronize_iwaitlist" "test_synchronize_event" "test_synchronize_event_iwait" "test_synchronize_event_iwaitlist")
dir=$(cd `dirname $0`; pwd)
echo $dir
logdir=$dir/logs
if [ ! -d $logdir ]; then
    mkdir $logdir
fi
datestr=$(date +%m%d)
echo $datestr
for func in ${funcs[@]}; do
  echo $func
  python -m bagua.distributed.launch --nproc_per_node=4 test.py --func $func > $logdir/test-$func-$datestr.log
done
