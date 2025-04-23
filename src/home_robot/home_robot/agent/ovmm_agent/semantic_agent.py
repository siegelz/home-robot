# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import torch

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent, Skill
from home_robot.core.interfaces import DiscreteNavigationAction, Observations


class SemanticAgent(OpenVocabManipAgent):
    """
    An agent that extends the OpenVocabManipAgent with custom navigation and exploration
    capabilities. Currently, it uses the parent class's navigation methods.
    """

    def __init__(self, config, device_id: int = 0):
        super().__init__(config, device_id=device_id)
        print("Initialized the semantic agent with OVMM capabilities!")
        
        # Add any custom initialization here
        self.custom_nav_params = getattr(config.AGENT, "CUSTOM_NAV", None)
        
    def _nav_to_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """
        Custom navigation to object implementation.
        Currently just calls the parent class implementation.
        """
        # For now, just use the parent class implementation
        return super()._nav_to_obj(obs, info)
    
    def _nav_to_rec(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """
        Custom navigation to receptacle implementation.
        Currently just calls the parent class implementation.
        """
        # For now, just use the parent class implementation
        return super()._nav_to_rec(obs, info)
    
    def _explore(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """
        Custom exploration implementation.
        This method would be called if the agent is in the EXPLORE state.
        """
        # For now, use a simple implementation that calls the navigation method
        return self._nav_to_obj(obs, info)
    
    def reset_vectorized(self):
        """Initialize agent state."""
        super().reset_vectorized()
        # Add any custom reset logic here
        
    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        super().reset_vectorized_for_env(e)
        # Add any custom reset logic here
        
    # You can add more custom methods here as needed
