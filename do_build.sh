#!/usr/bin/env bash
set -e

IMAGE_NAME="cidc25_casupnet_l1:latest"

docker build -t ${IMAGE_NAME} .
