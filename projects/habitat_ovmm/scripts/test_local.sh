#!/usr/bin/env bash

DOCKER_NAME="ovmm_baseline_submission"
SPLIT="minival"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
      --split)
      shift
      SPLIT="${1}"
      shift
      ;;
    *)
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run \
      -v $(realpath ../../data):/home-robot/data \
      -v $(realpath ../../src/home_robot/home_robot/agent):/home-robot/src/home_robot/home_robot/agent \
      -v /data/gibson:/Object-Goal-Navigation/data/scene_datasets/gibson_semantic \
      -v /data/objectnav:/Object-Goal-Navigation/data/datasets/objectnav \
      --runtime=nvidia \
      --gpus all \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "LOCAL_ARGS='habitat.dataset.split=${SPLIT}'" \
      ${DOCKER_NAME}
