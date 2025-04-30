# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
import gym

from home_robot.navigation_policy.object_navigation.objectnav_frontier_exploration_policy import ObjectNavFrontierExplorationPolicy
from home_robot.ogn.model import RL_Policy
from home_robot.ogn.utils.storage import GlobalRolloutStorage


class ObjectNavSemanticExplorationPolicy(ObjectNavFrontierExplorationPolicy):
    """
    Policy to select high-level goals for Object Goal Navigation that uses semantic information
    to guide exploration when the target object is not visible.
    
    This policy builds on the frontier exploration by adding semantic-aware exploration
    when the object goal is not found in the map, based on RL_policy from OGN paper
    """

    def __init__(
        self,
        exploration_strategy: str,
        num_sem_categories: int,
        explored_area_dilation_radius=10,
    ):
        super().__init__(exploration_strategy, num_sem_categories, explored_area_dilation_radius)

        # Initialize semantic navigation policy network
        cuda = True
        self.device = torch.device("cuda:0" if cuda else "cpu")

        # arguments (taken from OGN arguments.py)
        self.g_hidden_size = 256 # default global hidden size
        self.use_recurrent_global = 0 # default: no
        self.num_sem_categories = 16
        self.map_size_cm = 2400
        self.map_resolution = 5
        self.global_downscaling = 2
        self.num_global_steps = 20
        self.num_scenes = 1 # num_processes but we just want 1 for now. TODO check later?

        g_policy_load = "/home-robot/data/ogn/sem_exp.pth"

        # Calculating full and local map sizes
        # nc = self.num_sem_categories + 4  # num channels
        map_size = self.map_size_cm // self.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / self.global_downscaling)
        self.local_h = int(self.full_h / self.global_downscaling)

        # Global policy observation space
        self.ngc = 8 + self.num_sem_categories  # 24
        self.es = 2 # extras size
        self.g_observation_space = gym.spaces.Box(0, 1,
                                         (self.ngc,
                                          self.local_w,
                                          self.local_h), dtype='uint8')

        # Global policy action space
        self.g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                    shape=(2,), dtype=np.float32)

        # Global policy network for navigation
        self.g_policy = RL_Policy(self.g_observation_space.shape, self.g_action_space,
                         model_type=1,
                         base_kwargs={'recurrent': self.use_recurrent_global,
                                      'hidden_size': self.g_hidden_size,
                                      'num_sem_categories': self.ngc - 8
                                      }).to(self.device)

        # Storage (TODO do we need this? if just using inference, maybe we don't)
        # self.g_rollouts = GlobalRolloutStorage(self.num_global_steps,
        #                               self.num_scenes, self.g_observation_space.shape,
        #                               self.g_action_space, self.g_policy.rec_state_size,
        #                               self.es).to(self.device)
        # only need to keep track of the last one, so we dont' need num_global_steps
        self.global_input = torch.zeros(self.num_scenes, *self.g_observation_space.shape).to(self.device)
        self.g_rec_states = torch.zeros(self.num_scenes, self.g_policy.rec_state_size).to(self.device)
        self.masks = torch.ones(self.num_scenes)  # i think this is just a 1 lol, bc num_scenes is 1
        self.extras = torch.zeros(self.num_scenes, self.es, dtype=torch.long).to(self.device)

        '''
        '''

        print("Loading model {} for inference".format(g_policy_load))
        state_dict = torch.load(g_policy_load, map_location=lambda storage, loc: storage)
        self.g_policy.load_state_dict(state_dict)
        self.g_policy.eval()

    def forward(
        self,
        global_map, # 22 channels. we only need 20
        local_map,
        local_pose,
        map_features,
        object_category=None,
        start_recep_category=None,
        end_recep_category=None,
        instance_id=None,
        nav_to_recep=None,
    ):

        # TODO COMBINE local map and full_map (downsampled). 
        # breakpoint()
        global_input = torch.zeros(1, self.ngc, self.local_w, self.local_h).to(self.device)
        # Fill in map channels
        global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
        global_input[:, 4:8, :, :] = torch.nn.MaxPool2d(self.global_downscaling)(
            global_map[:, 0:4, :, :]) # is global_map same as full_map?
        # note that this ignores 5,6 in global_map and local_map, that is fine we don't need it
        global_input[:, 8:, :, :] = local_map[:, 6:, :, :].detach()

        # breakpoint()

        self.global_input = global_input
        self.local_pose = local_pose
        self.object_category = object_category
        self.start_recep_category = start_recep_category
        return super().forward(
            map_features,
            object_category,
            start_recep_category,
            end_recep_category,
            instance_id,
            nav_to_recep
        )

    def _calculate_extras(self):
        # prepare extras
        locs = self.local_pose.cpu().numpy()
        global_orientation = torch.zeros(self.num_scenes, 1).long()
        for e in range(self.num_scenes):
            global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
        self.extras[:, 0] = global_orientation[:, 0]
        # self.extras[:, 1] = self.object_category
        self.extras[:, 1] = self.start_recep_category # we want to be navigating to the start recep, not the object (find where a table is likely to be, not where a cup is likely to be)

    def explore_otherwise(self, map_features, goal_map, found_goal):
        """
        Override to use semantic information to guide exploration when the object goal
        is not found in the map.
        
        Args:
            map_features: semantic map features (shape[0] is batch_size/num_scenes)
            goal_map: binary map encoding goal(s) as they are currently
            found_goal: binary variables denoting whether we found the goal
            
        Returns:
            goal_maps: updated binary map encoding goal(s), for entire batch
        """
        print("===== EXPLORE OTHERWISE =====")

        # breakpoint() # check what map_features looks like?
        num_scenes = map_features.shape[0] # (1, 17, 480, 480)

        # Run Global Policy (global_goals = Long-Term Goal)
        self._calculate_extras()
        g_value, g_action, g_action_log_prob, self.g_rec_states = \
            self.g_policy.act(
                self.global_input, 
                self.g_rec_states,
                self.masks,
                extras=self.extras,
                deterministic=False
            )

        # TODO how to update masks with termination information? would have to pass it all the way back down
        # TODO update extras?

        # g_action is 2D BOX [0,1] x [0,1]
        cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
        global_goals = [[int(action[0] * self.local_w), int(action[1] * self.local_h)]
                        for action in cpu_actions] # [(4, 5)]
        global_goals = [[min(x, int(self.local_w - 1)), min(y, int(self.local_h - 1))]
                        for x, y in global_goals]

        # print("======= GLOBAL GOALS =========")
        # print(global_goals)

        # only update if not found goal
        # print("==> FOUND GOAL REC in explore otherwise", found_goal)
        for e in range(num_scenes):
            if not found_goal[e]:
                print("==== USING OGN ====")
                goal_map[e, global_goals[e][0], global_goals[e][1]] = 1

        return goal_map