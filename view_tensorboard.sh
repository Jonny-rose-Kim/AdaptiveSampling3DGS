#!/bin/bash
# TensorBoard로 두 모델의 metrics 비교

tensorboard --logdir_spec \
  adaptive:output/museum_adaptive/museum_adaptive,\
  original:output/museum_original/museum_original \
  --port 6006
