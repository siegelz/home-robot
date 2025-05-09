#!/usr/bin/env python3

import argparse
import gzip
import json
import os
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Set, Tuple

# COCO categories of interest
COCO_CATEGORIES = {
    "chair",
    "couch",
    "potted plant",
    "bed",
    "toilet",
    "tv",
    "dining-table",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "cup",
    "bottle"
}

# Mapping from OVMM categories to COCO categories
OVMM_TO_COCO_MAPPING = {
    # Direct matches for objects
    "book": "book",
    "bottle": "bottle",
    "clock": "clock",
    "cup": "cup",
    "vase": "vase",
    
    # Approximate matches for objects
    "plant_container": "potted plant",
    "plant_saucer": "potted plant",
    "glass": "cup",
    "jug": "bottle",
    "spray_bottle": "bottle",
    
    # Direct matches for receptacles
    "chair": "chair",
    "couch": "couch",
    "bed": "bed",
    "toilet": "toilet",
    "sink": "sink",
    "table": "dining-table",
    
    # Approximate matches for receptacles
    "counter": "dining-table",
    "stool": "chair",
    "bench": "chair",
}

def load_episodes(split: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load episodes from the specified split.
    
    Args:
        split: The dataset split (e.g., 'train', 'val', 'minival')
        
    Returns:
        Tuple of (full dataset dict, list of episode dictionaries)
    """
    episodes_path = f"data/datasets/ovmm/{split}/episodes.json.gz"
    
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")
    
    with gzip.open(episodes_path, 'rt') as f:
        data = json.load(f)
    
    return data, data["episodes"]

def analyze_episodes(episodes: List[Dict[str, Any]], coco_only: bool = False) -> Dict[str, Any]:
    """
    Analyze the distribution of categories in the episodes.
    
    Args:
        episodes: List of episode dictionaries
        coco_only: If True, only count categories that map to COCO categories
        
    Returns:
        Dictionary with analysis results
    """
    object_categories = Counter()
    start_recep_categories = Counter()
    end_recep_categories = Counter()
    coco_episodes = []
    
    for episode in episodes:
        obj_cat = episode.get("object_category", "")
        start_cat = episode.get("start_recep_category", "")
        end_cat = episode.get("goal_recep_category", "")
        
        # Count all categories
        object_categories[obj_cat] += 1
        start_recep_categories[start_cat] += 1
        end_recep_categories[end_cat] += 1
        
        # Check if this episode has COCO categories
        if coco_only:
            obj_is_coco = obj_cat in OVMM_TO_COCO_MAPPING
            start_is_coco = start_cat in OVMM_TO_COCO_MAPPING
            end_is_coco = end_cat in OVMM_TO_COCO_MAPPING
            
            if obj_is_coco and start_is_coco and end_is_coco:
                # Store the full episode for potential saving
                coco_episodes.append(episode)
    
    result = {
        "object_categories": object_categories,
        "start_recep_categories": start_recep_categories,
        "end_recep_categories": end_recep_categories
    }
    
    if coco_only:
        result["coco_episodes"] = coco_episodes
        result["coco_object_categories"] = Counter([ep.get("object_category", "") for ep in coco_episodes])
        result["coco_start_categories"] = Counter([ep.get("start_recep_category", "") for ep in coco_episodes])
        result["coco_end_categories"] = Counter([ep.get("goal_recep_category", "") for ep in coco_episodes])
    
    return result

def collect_referenced_indices(episodes: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    """
    Collect all transformation and viewpoint indices referenced in the episodes.
    
    Args:
        episodes: List of episode dictionaries
        
    Returns:
        Tuple of (transformation indices, viewpoint indices)
    """
    transformation_indices = set()
    viewpoint_indices = set()
    
    for episode in episodes:
        # Collect transformation indices from rigid_objs
        for rigid_obj in episode.get("rigid_objs", []):
            if isinstance(rigid_obj, list) and len(rigid_obj) >= 2:
                transform_idx = rigid_obj[1]
                if isinstance(transform_idx, int):
                    transformation_indices.add(transform_idx)
        
        # Collect viewpoint indices from candidate objects and receptacles
        for field in ["candidate_objects", "candidate_objects_hard", "candidate_start_receps", "candidate_goal_receps"]:
            for obj in episode.get(field, []):
                for view_idx in obj.get("view_points", []):
                    if isinstance(view_idx, int):  # It's an index, not already deserialized
                        viewpoint_indices.add(view_idx)
    
    return sorted(transformation_indices), sorted(viewpoint_indices)

def create_index_mappings(old_indices: List[int]) -> Dict[int, int]:
    """
    Create a mapping from old indices to new consecutive indices.
    
    Args:
        old_indices: List of old indices
        
    Returns:
        Dictionary mapping old indices to new indices
    """
    return {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}

def update_episode_references(episode: Dict[str, Any], transform_mapping: Dict[int, int], viewpoint_mapping: Dict[int, int]) -> Dict[str, Any]:
    """
    Update the indices in the episode to reference the new matrices.
    
    Args:
        episode: Episode dictionary
        transform_mapping: Mapping from old to new transformation indices
        viewpoint_mapping: Mapping from old to new viewpoint indices
        
    Returns:
        Updated episode dictionary
    """
    # Create a deep copy to avoid modifying the original
    updated_episode = episode.copy()
    
    # Update rigid_objs references
    if "rigid_objs" in updated_episode:
        new_rigid_objs = []
        for rigid_obj in updated_episode["rigid_objs"]:
            if isinstance(rigid_obj, list) and len(rigid_obj) >= 2:
                obj_handle = rigid_obj[0]
                transform_idx = rigid_obj[1]
                if isinstance(transform_idx, int) and transform_idx in transform_mapping:
                    new_transform_idx = transform_mapping[transform_idx]
                    new_rigid_objs.append([obj_handle, new_transform_idx])
                else:
                    new_rigid_objs.append(rigid_obj)
            else:
                new_rigid_objs.append(rigid_obj)
        updated_episode["rigid_objs"] = new_rigid_objs
    
    # Update viewpoint references in candidate objects and receptacles
    for field in ["candidate_objects", "candidate_objects_hard", "candidate_start_receps", "candidate_goal_receps"]:
        if field in updated_episode:
            for obj in updated_episode[field]:
                if "view_points" in obj:
                    new_view_points = []
                    for view_idx in obj["view_points"]:
                        if isinstance(view_idx, int) and view_idx in viewpoint_mapping:
                            new_view_idx = viewpoint_mapping[view_idx]
                            new_view_points.append(new_view_idx)
                        else:
                            # Keep as is if not in mapping or not an integer
                            new_view_points.append(view_idx)
                    obj["view_points"] = new_view_points
    
    return updated_episode

def update_viewpoint_references(episode: Dict[str, Any], viewpoint_mapping: Dict[int, int]) -> Dict[str, Any]:
    """
    Update only the viewpoint indices in the episode to reference the new matrices.
    
    Args:
        episode: Episode dictionary
        viewpoint_mapping: Mapping from old to new viewpoint indices
        
    Returns:
        Updated episode dictionary
    """
    # Create a deep copy to avoid modifying the original
    updated_episode = episode.copy()
    
    # Update viewpoint references in candidate objects and receptacles
    for field in ["candidate_objects", "candidate_objects_hard", "candidate_start_receps", "candidate_goal_receps"]:
        if field in updated_episode:
            for obj in updated_episode[field]:
                if "view_points" in obj:
                    new_view_points = []
                    for view_idx in obj["view_points"]:
                        if isinstance(view_idx, int) and view_idx in viewpoint_mapping:
                            new_view_idx = viewpoint_mapping[view_idx]
                            new_view_points.append(new_view_idx)
                        else:
                            # Keep as is if not in mapping or not an integer
                            new_view_points.append(view_idx)
                    obj["view_points"] = new_view_points
    
    return updated_episode

def save_coco_split(original_split: str, coco_episodes: List[Dict[str, Any]], full_dataset: Dict[str, Any], num_episodes: int = None) -> None:
    """
    Save COCO-compatible episodes to a new split file.
    
    Args:
        original_split: The name of the original split
        coco_episodes: List of episodes with COCO categories
        full_dataset: The full dataset dictionary
        num_episodes: Number of episodes to save (if None, save all)
    """
    if not coco_episodes:
        print(f"No COCO-compatible episodes found in {original_split} split. Nothing to save.")
        return
    
    # Limit the number of episodes if specified
    if num_episodes is not None and num_episodes > 0:
        if num_episodes < len(coco_episodes):
            coco_episodes = coco_episodes[:num_episodes]
            print(f"Limiting to {num_episodes} episodes")
    
    # Create the new split name and directory
    suffix = f"_coco{num_episodes if num_episodes is not None else ''}"
    new_split = f"{original_split}{suffix}"
    new_split_dir = f"data/datasets/ovmm/{new_split}"
    os.makedirs(new_split_dir, exist_ok=True)
    
    try:
        # Paths for original files
        transformations_path = f"data/datasets/ovmm/{original_split}/transformations.npy"
        viewpoints_path = f"data/datasets/ovmm/{original_split}/viewpoints.npy"
        
        if os.path.exists(transformations_path) and os.path.exists(viewpoints_path):
            # Simply copy the transformations.npy file (it's identical across splits)
            import shutil
            shutil.copy2(transformations_path, f"{new_split_dir}/transformations.npy")
            print(f"Copied transformations.npy from {original_split} to {new_split}")
            
            # For viewpoints, we still need to extract the specific ones used
            original_viewpoints = np.load(viewpoints_path)
            
            # Collect viewpoint indices
            _, viewpoint_indices = collect_referenced_indices(coco_episodes)
            
            # Create mapping
            viewpoint_mapping = create_index_mappings(viewpoint_indices)
            
            # Extract needed viewpoints
            if viewpoint_indices:
                new_viewpoints = np.array([original_viewpoints[idx] for idx in viewpoint_indices])
            else:
                # Create empty array with same shape except first dimension
                new_viewpoints = np.zeros((0,) + original_viewpoints.shape[1:], dtype=original_viewpoints.dtype)
            
            # Update episode references (only for viewpoints)
            updated_episodes = []
            for episode in coco_episodes:
                # Only update viewpoint references, not transformation references
                updated_episode = update_viewpoint_references(episode, viewpoint_mapping)
                updated_episodes.append(updated_episode)
            
            # Create a new dataset dictionary
            new_dataset = full_dataset.copy()
            new_dataset["episodes"] = updated_episodes
            
            # Save the new files
            np.save(f"{new_split_dir}/viewpoints.npy", new_viewpoints)
            
            with gzip.open(f"{new_split_dir}/episodes.json.gz", 'wt') as f:
                json.dump(new_dataset, f)
            
            print(f"\nCreated new split '{new_split}' with {len(updated_episodes)} episodes")
            print(f"Saved to {new_split_dir}")
            print(f"  - transformations.npy: copied from {original_split}")
            print(f"  - viewpoints.npy: {len(viewpoint_indices)} viewpoints")
        else:
            # If matrices don't exist, just save the episodes
            print(f"Warning: Could not find transformation or viewpoint matrices for {original_split} split.")
            print(f"Creating {new_split} with episodes only, without matrices.")
            
            # Create a new dataset dictionary
            new_dataset = full_dataset.copy()
            new_dataset["episodes"] = coco_episodes
            
            with gzip.open(f"{new_split_dir}/episodes.json.gz", 'wt') as f:
                json.dump(new_dataset, f)
            
            print(f"\nCreated new split '{new_split}' with {len(coco_episodes)} episodes")
            print(f"Saved to {new_split_dir}")
    
    except Exception as e:
        print(f"Error saving COCO split: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract episode information from OVMM dataset")
    parser.add_argument("--split", type=str, default="minival", 
                        help="Dataset split (e.g., train, val, minival)")
    parser.add_argument("--coco", action="store_true",
                        help="Filter for episodes with COCO categories only")
    parser.add_argument("--list-all-categories", action="store_true",
                        help="List all unique categories for mapping purposes")
    parser.add_argument("--save_split", action="store_true",
                        help="Save COCO-filtered episodes as a new split (requires --coco)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Number of episodes to save in the new split (default: all)")
    args = parser.parse_args()
    
    # Validate arguments
    if args.save_split and not args.coco:
        print("Error: --save_split can only be used with --coco")
        return
    
    try:
        # Load episodes
        full_dataset, episodes = load_episodes(args.split)
        print(f"Loaded {len(episodes)} episodes from {args.split} split")
        
        # Special mode to list all unique categories
        if args.list_all_categories:
            object_categories = set()
            start_recep_categories = set()
            end_recep_categories = set()
            
            for episode in episodes:
                object_categories.add(episode.get("object_category", ""))
                start_recep_categories.add(episode.get("start_recep_category", ""))
                end_recep_categories.add(episode.get("goal_recep_category", ""))
            
            print("\nAll unique object categories:")
            for category in sorted(object_categories):
                print(f'    "{category}": "",')
            
            print("\nAll unique receptacle categories:")
            all_recep = sorted(start_recep_categories.union(end_recep_categories))
            for category in all_recep:
                print(f'    "{category}": "",')
            
            return
        
        # Normal analysis mode
        analysis = analyze_episodes(episodes, args.coco)
        
        print("\nObject category distribution:")
        for category, count in analysis["object_categories"].most_common():
            print(f"  {category}: {count}")
        
        print("\nStart receptacle category distribution:")
        for category, count in analysis["start_recep_categories"].most_common():
            print(f"  {category}: {count}")
        
        print("\nEnd receptacle category distribution:")
        for category, count in analysis["end_recep_categories"].most_common():
            print(f"  {category}: {count}")
        
        if args.coco:
            coco_episodes = analysis["coco_episodes"]
            print(f"\nFound {len(coco_episodes)} episodes with COCO categories")
            
            print("\nCOCO object category distribution:")
            for category, count in analysis["coco_object_categories"].most_common():
                print(f"  {category}: {count}")
            
            print("\nCOCO start receptacle category distribution:")
            for category, count in analysis["coco_start_categories"].most_common():
                print(f"  {category}: {count}")
            
            print("\nCOCO end receptacle category distribution:")
            for category, count in analysis["coco_end_categories"].most_common():
                print(f"  {category}: {count}")
            
            # Save the COCO split if requested
            if args.save_split:
                save_coco_split(args.split, coco_episodes, full_dataset, args.num_episodes)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
