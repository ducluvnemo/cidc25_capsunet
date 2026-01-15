#!/usr/bin/env bash
set -e

IMAGE_NAME="cidc25_casupnet_l1:latest"
OUT_FILE="cidc25_casupnet_l1.tar.gz"

./do_build.sh

docker save ${IMAGE_NAME} | gzip > ${OUT_FILE}
echo "Saved container to ${OUT_FILE}"
