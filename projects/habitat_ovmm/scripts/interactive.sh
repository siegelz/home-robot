#!/usr/bin/env bash

DOCKER_NAME="ovmm_raz_submission"
SPLIT="val_coco30"

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
      --baseline_config_path)
      shift
      BASELINE_CONFIG_PATH="${1}"
      shift
      ;;
    *)
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run -it --entrypoint /bin/bash \
      -v $(realpath ../../data):/home-robot/data \
      -v $(realpath ../../datadump):/home-robot/datadump \
      -v $(realpath ../../video_dir):/home-robot/video_dir \
      -v $(realpath ../../src/home_robot/home_robot/agent):/home-robot/src/home_robot/home_robot/agent \
      -v $(realpath ../../src/home_robot/home_robot/ogn):/home-robot/src/home_robot/home_robot/ogn \
      -v $(realpath ../../src/home_robot/home_robot/perception/constants.py):/home-robot/src/home_robot/home_robot/perception/constants.py \
      -v $(realpath ../../src/home_robot/home_robot/perception/wrapper.py):/home-robot/src/home_robot/home_robot/perception/wrapper.py \
      -v $(realpath ../../src/home_robot/home_robot/perception/detection/detic/detic_perception.py):/home-robot/src/home_robot/home_robot/perception/detection/detic/detic_perception.py \
      -v $(realpath ../../src/home_robot/home_robot/navigation_policy):/home-robot/src/home_robot/home_robot/navigation_policy \
      -v $(realpath ../../src/home_robot_sim/home_robot_sim/env):/home-robot/src/home_robot_sim/home_robot_sim/env \
      -v $(realpath ../../projects/habitat_ovmm/configs):/home-robot/projects/habitat_ovmm/configs \
      -v $(realpath ../../projects/habitat_ovmm/evaluator.py):/home-robot/projects/habitat_ovmm/evaluator.py \
      -v $(realpath ../../projects/habitat_ovmm/scripts/submission.sh):/home-robot/projects/habitat_ovmm/scripts/submission.sh \
      -v /data/gibson:/Object-Goal-Navigation/data/scene_datasets/gibson_semantic \
      -v /data/objectnav:/Object-Goal-Navigation/data/datasets/objectnav \
      --runtime=nvidia \
      --gpus all \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "BASELINE_CONFIG_PATH=${BASELINE_CONFIG_PATH}" \
      -e "LOCAL_ARGS=habitat.dataset.split=${SPLIT}" \
	${DOCKER_NAME}
