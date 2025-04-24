#!/usr/bin/env bash

python projects/habitat_ovmm/agent.py --evaluation_type $AGENT_EVALUATION_TYPE --baseline_config_path $BASELINE_CONFIG_PATH $@
