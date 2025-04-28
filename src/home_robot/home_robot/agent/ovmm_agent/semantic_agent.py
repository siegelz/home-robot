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

        breakpoint()

        # Store map state
        self.local_map = None
        self.local_pose = None
        self.global_map = None  # Full map
        self.planner_pose_inputs = None
        self.origins = None
        self.lmb = None  # Local map boundaries
        
        # Store dimensions
        self.local_w = None
        self.local_h = None
        self.ngc = None  # Number of global channels
        self.success_dist = config.AGENT.success_distance

    def _init_ogn_modules(self):
        """Adapted from OGN's main.py"""
        args = get_args(['--eval', '1', '--load', '/home-robot/data/ogn/sem_exp.pth', '--total_num_scenes', '1', '--sem_gpu_id', '0'])
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
        
    def _prepare_obs_for_ogn(self, obs: Observations) -> torch.Tensor:
        """
        Convert the observation object that HomeRobot uses into 
        the format used by the semantic navigation paper.
        
        Args:
            obs (Observations): HomeRobot observation object
            
        Returns:
            torch.Tensor: Observation tensor in the format expected by OGN (C, H, W)
        """
        # Extract RGB and depth from the Observations object
        rgb = obs.rgb  # Shape: (H, W, 3)
        depth = obs.depth  # Shape: (H, W)
        
        # Ensure RGB is in the correct format (uint8)
        rgb = rgb.astype(np.uint8)
        
        # Process depth similar to _preprocess_depth in Sem_Exp_Env_Agent
        depth_processed = self._preprocess_depth(depth, min_d=0.5, max_d=5.0)
        
        # Get semantic predictions using the same method as in sem_exp.py
        sem_seg_pred = self._get_sem_pred(rgb)
        
        # Expand depth to have a channel dimension
        depth_processed = np.expand_dims(depth_processed, axis=2)
        
        # Concatenate RGB, processed depth, and semantic predictions along the channel dimension
        combined = np.concatenate((rgb, depth_processed, sem_seg_pred), axis=2)  # Shape: (H, W, 4+16)
        
        # Transpose to get (C, H, W) format as expected by the model
        state = combined.transpose(2, 0, 1)  # Shape: (20, H, W)
        
        # Convert to PyTorch tensor
        state_tensor = torch.from_numpy(state).float()
        
        return state_tensor
        
    def _preprocess_depth(self, depth, min_d, max_d):
        """
        Process depth information to match the format expected by the model.
        This is the same as the _preprocess_depth method in Sem_Exp_Env_Agent.
        
        Args:
            depth (np.ndarray): Raw depth array of shape (H, W) or (H, W, 1)
            min_d (float): Minimum depth value
            max_d (float): Maximum depth value
            
        Returns:
            np.ndarray: Processed depth array of shape (H, W)
        """
        # Make sure we're working with a 2D array
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        else:
            depth = depth.copy()
        
        # Apply the same processing as in Sem_Exp_Env_Agent._preprocess_depth
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()
        
        mask2 = depth > 0.99
        depth[mask2] = 0.
        
        mask1 = depth == 0
        depth[mask1] = 100.0
        
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
        
    def _get_sem_pred(self, rgb):
        """
        Get semantic predictions for the RGB image.
        This uses the same approach as Sem_Exp_Env_Agent._get_sem_pred.
        
        Args:
            rgb (np.ndarray): RGB image of shape (H, W, 3)
            
        Returns:
            np.ndarray: Semantic predictions of shape (H, W, 16)
        """
        # Get semantic predictions using the semantic prediction model
        semantic_pred, _ = self.sem_pred.get_prediction(rgb)
        semantic_pred = semantic_pred.astype(np.float32)
        
        return semantic_pred

    # TODO OVERRIDE
    def _nav_to_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """
        Custom navigation to object implementation.
        Currently just calls the parent class implementation.
        Obs is already populated with semantic information
        """
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
        """
        Use the OGN semantic navigation to generate navigation actions.
        Returns: action, info, terminate
        """
        # Get observation in OGN format
        ogn_obs = self._prepare_obs_for_ogn(obs)
        
        # Update the semantic map
        with torch.no_grad():
            # Update map with current observation
            _, self.local_map, _, self.local_pose = self.sem_map_module(
                ogn_obs['rgbd'], 
                ogn_obs['pose'], 
                self.local_map, 
                self.local_pose
            )
        
        # Prepare global policy input
        global_input = torch.zeros(1, self.ngc, self.local_w, self.local_h).to(self.device)
        
        # Fill in map channels
        global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :].detach()
        global_input[:, 4:8, :, :] = self.global_map[:, 0:4, :, :]  # Assuming you've defined global_map
        global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()
        
        # Get goal category from task observations
        goal_cat_id = torch.tensor([obs.task_observations['goal_cat_id']]).to(self.device)
        
        # Get orientation
        orientation = torch.tensor([
            int((self.local_pose[0, 2] + 180.0) / 5.)
        ]).to(self.device)
        
        # Prepare extras for policy
        extras = torch.zeros(1, 2).to(self.device)
        extras[:, 0] = orientation
        extras[:, 1] = goal_cat_id
        
        # Act using global policy (get long-term goal)
        with torch.no_grad():
            _, g_action, _, _ = self.g_policy.act(
                global_input,
                self.g_rollouts.rec_states[0],
                torch.ones(1).to(self.device),  # Masks
                extras=extras,
                deterministic=True  # Use deterministic actions during evaluation
            )
        
        # Convert policy output to goal location
        cpu_actions = torch.sigmoid(g_action).cpu().numpy()
        global_goal = [
            int(cpu_actions[0, 0] * self.local_w), 
            int(cpu_actions[0, 1] * self.local_h)
        ]
        global_goal = [
            min(global_goal[0], int(self.local_w - 1)), 
            min(global_goal[1], int(self.local_h - 1))
        ]
        
        # Create goal map for local policy
        goal_map = np.zeros((self.local_w, self.local_h))
        goal_map[global_goal[0], global_goal[1]] = 1
        
        # Check if object is detected in map and prioritize it
        found_goal = False
        goal_cat_id_int = int(goal_cat_id.item())
        cn = goal_cat_id_int + 4  # Category channel in map
        if self.local_map[0, cn, :, :].sum() != 0:
            cat_semantic_map = self.local_map[0, cn, :, :].cpu().numpy()
            cat_semantic_scores = cat_semantic_map.copy()
            cat_semantic_scores[cat_semantic_scores > 0] = 1.0
            goal_map = cat_semantic_scores
            found_goal = True
        
        # Prepare planner input (for local policy/path planning)
        planner_input = {
            'map_pred': self.local_map[0, 0, :, :].cpu().numpy(),
            'exp_pred': self.local_map[0, 1, :, :].cpu().numpy(),
            'pose_pred': self.planner_pose_inputs[0],  # You need to maintain this
            'goal': goal_map,
            'found_goal': found_goal
        }
        
        # Local planning to get next action (you'll need to implement this)
        action = self._local_plan(planner_input)
        
        # Determine if we should terminate this skill
        # Terminate if we've found the goal and are close enough
        terminate = found_goal and np.linalg.norm(
            np.array(self.local_pose[0, :2].cpu()) - 
            np.array(global_goal)
        ) < self.success_dist
        
        # Update info with visualization data if needed
        info['sem_map_pred'] = self.local_map[0, 4:, :, :].argmax(0).cpu().numpy()
        info['goal_map'] = goal_map
        
        return action, info, terminate

    def _local_plan(self, planner_input: Dict) -> DiscreteNavigationAction:
        """
        Local planning to convert goal positions to discrete actions.
        This is a simplified version - you'll need to implement path planning.
        """
        # Implement simple local planning logic
        # For example, use FMM planning or a similar approach to get to the goal
        
        # For now, return a placeholder action
        return DiscreteNavigationAction.MOVE_FORWARD
    
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
