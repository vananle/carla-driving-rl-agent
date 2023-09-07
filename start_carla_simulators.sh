#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh --world-port=20000 -opengl
CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh --world-port=20002 -opengl
CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh --world-port=20004 -opengl
