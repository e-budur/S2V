#!/bin/bash
nohup tensorboard --logdir='/truba_scratch/ebudur/data/results' --port=8082 --host='0.0.0.0' > tensorboard.out 2>&1&
