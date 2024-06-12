#!/usr/bin/env bash

IMAGE_NAME="minisift-rust-build:latest"
WD=$(pwd)

echo "Rebuilding docker image"
# kill container if it's already running
docker build --rm -f run.Dockerfile . -t $IMAGE_NAME
# clean up dangling image, which could happen when dockerfile changes
docker rmi $(docker images -f "dangling=true" -q) > /dev/null 2>&1
echo "Done rebuilding docker image"

docker run \
  --rm \
  -v $WD:/root/minisift \
  -e OpenCV_DIR="/usr/local/lib/cmake/opencv4" \
  -e OpenCV_DIR="/usr/local/lib" \
  $IMAGE_NAME \
  /root/.cargo/bin/cargo run --release -- $1 $2