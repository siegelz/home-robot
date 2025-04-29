# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from home_robot.home_robot.navigation_policy.object_navigation.objectnav_frontier_exploration_policy import ObjectNavFrontierExplorationPolicy

class ObjectNavSemanticExplorationPolicy(ObjectNavFrontierExplorationPolicy):
    """
    Policy to select high-level goals for Object Goal Navigation that uses semantic information
    to guide exploration when the target object is not visible.
    
    This policy builds on the frontier exploration by adding semantic-aware exploration
    when the object goal is not found in the map, based on RL_policy from OGN paper
    """
    
    def explore_otherwise(self, map_features, goal_map, found_goal):
        """
        Override to use semantic information to guide exploration when the object goal
        is not found in the map.
        
        Args:
            map_features: semantic map features
            goal_map: binary map encoding goal(s)
            found_goal: binary variables denoting whether we found the goal
            
        Returns:
            goal_map: updated binary map encoding goal(s)
        """
        raise NotImplementedError()
        