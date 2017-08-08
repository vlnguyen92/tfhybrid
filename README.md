## Dependencies

- Python 2.x
- TensorFlow 0.12 r0 or higher
- tflearn (for `convnets/tflearn_alexnet.py`)
- matplotlib
- tmuxp (required to start local cluster conveniently): https://github.com/tony/tmuxp

## Project structure

### Python package `hybridalex`
This is the main package implementing the ATLAS library described in our report. The library is organized into several subpackages.

- `hybridalex.nets` contains the `ModelBuilder` and the implementation of Alexnet, VGG and Overfeat.
- `hybridalex.datasets` provides a unified interface to various datasets.
- `hybridalex.optimizers` contains a custom optimizer that is able to set learning_rate seperately on layer weights.
- `hybridalex.scripts` comes with the training scripts we used to produce the data in the report. This is the main entry point of the library.

### Scripts
We wrap python modules into shell scripts for easy execution. They are located in `scripts` folder from the top level.

#### Main training script
Located at `scripts/train_hybrid.sh`. This is the main training script that wee used to generate performance data for hybrid parallelism for 3 different networks.

The script requires a TensorFlow cluster accessable at `localhost:2222`, with one parameter server named `ps` and worker jobs named `worker`. `1dev` requires at least one worker job presenting in the cluster, `2dev-*` requires at least 2 and `4dev-*` requires at least 4.

The environment variable `CUDA_VISIBLE_DEVICES` must be set correctly when starting each of the server instance in the cluster, so that no GPU is visible to the parameter server, and at least one GPU is visible to each worker servers.

We provide a `tmuxp` session definition using which one can start local cluster with correct environment quickly. This requires `tmux` and `tmuxp` to be installed. Before using, please change the start directory in `hybridalex/clusterdef/localhost-2worker-1ps.yaml` accordingly.
```
tmuxp hybridalex/clusterdef/localhost-2worker-1ps.yaml
```

##### Dataset
The network can be trained with either randomly generated fake data or the Flowers dataset. The Flowers dataset must be downloaded first, which can be done with `scripts/download_flowers.sh`. Currently the data path is hard-coded to `~/data/flowers`.

##### Synopsis

```
usage: train_hybrid.sh [-h] [--network {alexnet,vgg,overfeat}]
                       [--mode {1dev,2dev-data,2dev-model,2dev-model-data,4dev-data,4dev-model,4dev-model-data}]
                       [--work_dir WORK_DIR] [--log_dir LOG_DIR]
                       [--model_dir MODEL_DIR] [--dataset {fake_data,flowers}]
                       [--batch_num BATCH_NUM] [--batch_size BATCH_SIZE]
                       [--redirect_outerr] [--eval]

optional arguments:
  -h, --help            show this help message and exit
  --network {alexnet,vgg,overfeat}
                        the name of the parallel method
  --mode {1dev,2dev-data,2dev-model,2dev-model-data,4dev-data,4dev-model,4dev-model-data}
                        the name of the parallel method
  --work_dir WORK_DIR   directory for saving files, defaults to
                        /tmp/workspace/tflogs
  --log_dir LOG_DIR     directory for tensorboard logs, defaults to
                        WORK_DIR/tf in train mode and WORK_DIR/tfeval in
                        evaluation mode
  --model_dir MODEL_DIR
                        directory for model checkpoints, defaults to
                        WORK_DIR/model
  --dataset {fake_data,flowers}
                        dataset to use
  --batch_num BATCH_NUM
                        total batch number
  --batch_size BATCH_SIZE
                        batch size
  --redirect_outerr     whether to redirect stdout to WORK_DIR/out.log and
                        stderr to WORK_DIR/err.log
  --eval                evaluation or train
```

#### Single layer inference benchmark
This script runs a forward pass through a single FC or CONV layer, with specified parallelism.
```
usage: train_single.sh [-h]
                       {fc_none,fc_model,fc_data,conv_none,conv_data,conv_model}

positional arguments:
  {fc_none,fc_model,fc_data,conv_none,conv_data,conv_model}

optional arguments:
  -h, --help            show this help message and exit
```

### Existing network implementations
The `convnets` folder contains our test scripts for existing neural network implementations.

`benchmark_*.py` are plain TensorFlow implementation using `tf.train.Supervisor`. They can only be used in a cluster and currently the IP addresses are hard-coded. So please change line 18-19 accordingly. After correctly specified the IP addresses of the cluster, the training can be started by launching the script on each server in the cluster.

```
python benchmark_alexnet.py --job_name ps --task_index 0  # on parameter server
python benchmark_alexnet.py --job_name worker --task_index 0  # on worker 0
python benchmark_alexnet.py --job_name worker --task_index 1  # on worker 1
```

`tflearn_alexnet.py` is the Alexnet implementation from TFLearn, a popular deep learning library built on top of TensorFlow. The script can be run as is.

### Microbenchmarks
The folder contains the code we used to learn TensorFlow. Most of them are just simple demo of TensorFlow and its cluster mode. They may not work as is.

### Logs
This is some of the most important raw log file we generated during the semester and from which the figures in the report are plotted.
