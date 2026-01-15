#!/usr/bin/env bash
set -e

IMAGE_NAME="cidc25_casupnet_l1:latest"

mkdir -p test/input/images
mkdir -p test/output/images/stacked-neuron-images-with-reduced-noise

# Copy 1 file TIFF noisy của CIDC25 vào test/input/images trước khi chạy.

./do_build.sh

docker run --rm \
  -v "$(pwd)/test/input:/input" \
  -v "$(pwd)/test/output:/output" \
  -e INPUT_PATH="/input/images" \
  -e OUTPUT_PATH="/output/images/stacked-neuron-images-with-reduced-noise" \
  cidc25_casupnet_l1:latest
