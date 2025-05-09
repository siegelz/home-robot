#!/usr/bin/env python3

import argparse
import gzip
import json
import os
from collections import Counter
from typing import Dict, List, Any, Set

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

def load_episodes(split: str) -> List[Dict[str, Any]]:
    """
    Load episodes from the specified split.
    
    Args:
        split: The dataset split (e.g., 'train', 'val', 'minival')
        
    Returns:
        List of episode dictionaries
    """
    episodes_path = f"data/datasets/ovmm/{split}/episodes.json.gz"
    
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")
    
    with gzip.open(episodes_path, 'rt') as f:
        data = json.load(f)
    
    return data["episodes"]

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
                coco_episodes.append({
                    "episode_id": episode.get("episode_id", ""),
                    "scene_id": episode.get("scene_id", "").split("/")[-1].split(".")[0],
                    "object_category": obj_cat,
                    "start_recep_category": start_cat,
                    "end_recep_category": end_cat,
                    "coco_object": OVMM_TO_COCO_MAPPING.get(obj_cat, ""),
                    "coco_start": OVMM_TO_COCO_MAPPING.get(start_cat, ""),
                    "coco_end": OVMM_TO_COCO_MAPPING.get(end_cat, "")
                })
    
    result = {
        "object_categories": object_categories,
        "start_recep_categories": start_recep_categories,
        "end_recep_categories": end_recep_categories
    }
    
    if coco_only:
        result["coco_episodes"] = coco_episodes
        result["coco_object_categories"] = Counter([ep["coco_object"] for ep in coco_episodes])
        result["coco_start_categories"] = Counter([ep["coco_start"] for ep in coco_episodes])
        result["coco_end_categories"] = Counter([ep["coco_end"] for ep in coco_episodes])
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Extract episode information from OVMM dataset")
    parser.add_argument("--split", type=str, default="minival", 
                        help="Dataset split (e.g., train, val, minival)")
    parser.add_argument("--coco", action="store_true",
                        help="Filter for episodes with COCO categories only")
    parser.add_argument("--list-all-categories", action="store_true",
                        help="List all unique categories for mapping purposes")
    args = parser.parse_args()
    
    try:
        episodes = load_episodes(args.split)
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
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
