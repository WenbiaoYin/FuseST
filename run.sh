#!/bin/bash

set -e
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

pip3 install -U pip -i https://bytedpypi.byted.org/simple
sudo pip install --editable ./ -i https://bytedpypi.byted.org/simple

Distributed_Train="python3 -m torch.distributed.launch \
    --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr="$METIS_WORKER_0_HOST" \
    --master_port=$METIS_WORKER_0_PORT train.py "

if [[ "$@" == bash* ]] || [[ "$@" == fairseq* ]];then
    $@
else
    if [ $ARNOLD_WORKER_NUM -gt 1 ];then
        $Distributed_Train "$@"
    else
        fairseq-train "$@"
    fi
fi