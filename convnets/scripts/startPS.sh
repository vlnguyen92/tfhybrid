#!/bin/bash
tmux new-session -d -s tf-ps 'CUDA_VISIBLE_DEVICES='' python ../benchmark_overfeat.py --job_name=ps --task_index=0'
tmux detach
