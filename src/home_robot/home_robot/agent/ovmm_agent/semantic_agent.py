# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent, Skill
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.ogn.agents.utils.semantic_prediction import SemanticPredMaskRCNN

from home_robot.ogn.model import RL_Policy, Semantic_Mapping 
from home_robot.ogn.utils.storage import GlobalRolloutStorage
from home_robot.ogn.arguments import get_args
import home_robot.ogn.algo as algo

import gym

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

        self._init_ogn_modules()

        print("Initialized the OGN modules for semantic navigation!")

    def _init_ogn_modules(self):
        """Adapted from OGN's main.py"""
        args = get_args(['--eval', '1', '--load', '/home-robot/data/ogn/sem_exp.pth', '--total_num_scenes', '1'])
        device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
        num_scenes = args.num_processes

        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        nc = args.num_sem_categories + 4  # num channels

        # Calculating full and local map sizes
        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w = int(full_w / args.global_downscaling)
        local_h = int(full_h / args.global_downscaling)

        # Initializing full and local map
        full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
        local_map = torch.zeros(num_scenes, nc, local_w,
                                local_h).float().to(device)

        # Initial full and local pose
        full_pose = torch.zeros(num_scenes, 3).float().to(device)
        local_pose = torch.zeros(num_scenes, 3).float().to(device)

        # Origin of local map
        origins = np.zeros((num_scenes, 3))

        # Local Map Boundaries
        lmb = np.zeros((num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        planner_pose_inputs = np.zeros((num_scenes, 7))

        def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
            loc_r, loc_c = agent_loc
            local_w, local_h = local_sizes
            full_w, full_h = full_sizes

            if args.global_downscaling > 1:
                gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
                gx2, gy2 = gx1 + local_w, gy1 + local_h
                if gx1 < 0:
                    gx1, gx2 = 0, local_w
                if gx2 > full_w:
                    gx1, gx2 = full_w - local_w, full_w

                if gy1 < 0:
                    gy1, gy2 = 0, local_h
                if gy2 > full_h:
                    gy1, gy2 = full_h - local_h, full_h
            else:
                gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

            return [gx1, gx2, gy1, gy2]

        def init_map_and_pose():
            full_map.fill_(0.)
            full_pose.fill_(0.)
            full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

            locs = full_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                (local_w, local_h),
                                                (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                            lmb[e][0] * args.map_resolution / 100.0, 0.]

            for e in range(num_scenes):
                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

        def init_map_and_pose_for_env(e):
            full_map[e].fill_(0.)
            full_pose[e].fill_(0.)
            full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

            locs = full_pose[e].cpu().numpy()
            planner_pose_inputs[e, :3] = locs
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                            (local_w, local_h),
                                            (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                        lmb[e][0] * args.map_resolution / 100.0, 0.]

            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

        def update_intrinsic_rew(e):
            prev_explored_area = full_map[e, 1].sum(1).sum(0)
            full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                local_map[e]
            curr_explored_area = full_map[e, 1].sum(1).sum(0)
            intrinsic_rews[e] = curr_explored_area - prev_explored_area
            intrinsic_rews[e] *= (args.map_resolution / 100.)**2  # to m^2

        init_map_and_pose()

        # Global policy observation space
        ngc = 8 + args.num_sem_categories
        es = 2
        g_observation_space = gym.spaces.Box(0, 1,
                                            (ngc,
                                            local_w,
                                            local_h), dtype='uint8')

        # Global policy action space
        g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                        shape=(2,), dtype=np.float32)

        # Global policy recurrent layer size
        g_hidden_size = args.global_hidden_size
        
        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(args).to(device)
        self.sem_map_module.eval()

        # Global policy
        self.g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                            model_type=1,
                            base_kwargs={'recurrent': args.use_recurrent_global,
                                        'hidden_size': g_hidden_size,
                                        'num_sem_categories': ngc - 8
                                        }).to(device)
        self.g_agent = algo.PPO(self.g_policy, args.clip_param, args.ppo_epoch,
                        args.num_mini_batch, args.value_loss_coef,
                        args.entropy_coef, lr=args.lr, eps=args.eps,
                        max_grad_norm=args.max_grad_norm)

        self.sem_pred = SemanticPredMaskRCNN(args)

        intrinsic_rews = torch.zeros(num_scenes).to(device)

        # Storage
        self.g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                        num_scenes, g_observation_space.shape,
                                        g_action_space, self.g_policy.rec_state_size,
                                        es).to(device)

        if args.load != "0":
            print("Loading model {}".format(args.load))
            state_dict = torch.load(args.load,
                                    map_location=lambda storage, loc: storage)
            self.g_policy.load_state_dict(state_dict)
        else:
            assert False, "Must load model"

        if args.eval:
            self.g_policy.eval()
        else:
            assert False, "Must be in eval mode"
        
    # TODO change the return type from any
    def _prepare_obs_for_ogn(self, obs: Observations) -> any:
        """
        Convert the observation object that HomeRobot uses into 
        the format used by the semantic navigation paper.
        """
        pass

    # TODO OVERRIDE
    def _nav_to_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """
        Custom navigation to object implementation.
        Currently just calls the parent class implementation.
        Obs is already populated with semantic information
        """
        # For now, just use the parent class implementation
        return super()._nav_to_obj(obs, info)
        # TODO
        if self.skip_skills.nav_to_obj:
            terminate = True
        if self.verbose:
            print("[SEMANTIC AGENT] semantic policy")
        action, info, terminate = self._ogn_nav(obs, info)

        new_state = None
        if terminate:
            action = None
            new_state = Skill.GAZE_AT_OBJ
        return action, info, new_state
    
    # TODO OVERRIDE (similar to nav to obj)
    def _nav_to_rec(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """
        Custom navigation to receptacle implementation.
        Currently just calls the parent class implementation.
        """
        # For now, just use the parent class implementation
        return super()._nav_to_rec(obs, info)

    def _ogn_nav(self, obs: Observations, info: Dict[str, Any]) -> Tuple[DiscreteNavigationAction, Any, bool]:
        # action, info, terminate = self._semantic_nav(obs, info)
        
        pass
    
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
