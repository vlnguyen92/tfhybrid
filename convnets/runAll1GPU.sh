#!/bin/bash
for f in *.py
do
    echo $f
    cmd_start="tmux new-session -d -s tf-ps \"CUDA_VISIBLE_DEVICES= python $f --job_name=ps --task_index=0\""
    eval $cmd_start
    job='python '$f' --job_name=worker --task_index=0'
    eval $job
    cmd_stop='tmux kill-session -t tf-ps'
    eval $cmd_stop
done
