#!/bin/bash
tmux new-session -d -s tf-worker 'python benchmark_alexnet.py --job_name=worker --task_index=0'
