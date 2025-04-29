# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.morphology import binary_dilation, binary_erosion

from home_robot.home_robot.navigation_policy.object_navigation.objectnav_frontier_exploration_policy import ObjectNavFrontierExplorationPolicy

class ObjectNavSemanticExplorationPolicy(ObjectNavFrontierExplorationPolicy):
    """
    Policy to select high-level goals for Object Goal Navigation that uses semantic information
    to guide exploration when the target object is not visible.
    
    This policy builds on the frontier exploration by adding semantic-aware exploration
    when the object goal is not found in the map.
    """
    
    def __init__(
        self,
        exploration_strategy: str,
        num_sem_categories: int, 
        explored_area_dilation_radius=10,
        semantic_weight=0.5
    ):
        super().__init__(
            exploration_strategy=exploration_strategy,
            num_sem_categories=num_sem_categories,
            explored_area_dilation_radius=explored_area_dilation_radius
        )
        # Weight to balance between frontier-based and semantic-based exploration
        self.semantic_weight = semantic_weight
    
    def get_semantic_score_map(self, map_features, object_category=None):
        """
        Create a score map that prioritizes exploration toward areas that are likely
        to contain the target object based on semantic correlations.
        
        Args:
            map_features: semantic map features (batch_size, channels, height, width)
            object_category: object goal category
            
        Returns:
            semantic_score_map: score map for semantic-guided exploration
        """
        batch_size, _, height, width = map_features.shape
        device = map_features.device
        semantic_score_map = torch.zeros((batch_size, height, width), device=device)
        
        if object_category is None:
            return semantic_score_map
            
        # Create a semantic score map based on semantic information
        # This is a simple implementation that could be enhanced with more
        # sophisticated semantic correlations between objects and locations
        for e in range(batch_size):
            # Calculate whether any semantic categories are visible in the map
            semantic_channels = map_features[e, 2*MC.NON_SEM_CHANNELS:2*MC.NON_SEM_CHANNELS+self.num_sem_categories, :, :]
            any_semantics = torch.any(semantic_channels > 0, dim=0).float()
            
            # Create distance map from existing semantic elements
            # This encourages exploration near already detected semantic elements
            if any_semantics.sum() > 0:
                # Convert to numpy for distance transform
                any_semantics_np = any_semantics.cpu().numpy()
                distance_map = scipy.ndimage.distance_transform_edt(1 - any_semantics_np)
                
                # Normalize and convert back to tensor
                if distance_map.max() > 0:
                    distance_map = 1.0 - (distance_map / distance_map.max())
                
                semantic_score_map[e] = torch.from_numpy(distance_map).to(device)
        
        return semantic_score_map
    
    def explore_otherwise(self, map_features, goal_map, found_goal, object_category=None):
        """
        Override to use semantic information to guide exploration when the object goal
        is not found in the map.
        
        Args:
            map_features: semantic map features
            goal_map: binary map encoding goal(s)
            found_goal: binary variables denoting whether we found the goal
            object_category: object goal category (optional)
            
        Returns:
            goal_map: updated binary map encoding goal(s)
        """
        # Get the frontier map (from parent class)
        frontier_map = self.get_frontier_map(map_features)
        
        # Get semantic score map
        semantic_score_map = self.get_semantic_score_map(map_features, object_category)
        
        # Combine frontier and semantic scores with weighted sum
        combined_score_map = (1 - self.semantic_weight) * frontier_map + self.semantic_weight * semantic_score_map
        
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                # Apply cluster filtering to focus on coherent regions
                score_map = combined_score_map[e]
                if score_map.sum() > 0:
                    # Normalize to create a proper probability distribution
                    score_map = score_map / score_map.max()
                    # Apply threshold to create binary goal map
                    goal_map[e] = (score_map > 0.5).float()
                    # If no points are above threshold, use highest scoring points
                    if goal_map[e].sum() == 0:
                        # Select top 1% of points
                        num_points = max(1, int(0.01 * score_map.numel()))
                        threshold = torch.topk(score_map.flatten(), num_points).values[-1]
                        goal_map[e] = (score_map >= threshold).float()
                else:
                    # Fallback to frontier exploration if no semantic score
                    goal_map[e] = frontier_map[e]
                
                # Apply cluster filtering to focus on largest coherent region
                if goal_map[e].sum() > 0:
                    goal_map[e] = self.cluster_filtering(goal_map[e])
        
        return goal_map
