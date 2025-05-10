#!/usr/bin/env python3
import os
import json
import csv
import glob
import math
from collections import defaultdict

def load_matches(file_path):
    """Load the matches.csv file that contains the selected (correct) matches"""
    matches = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            image_id = row[0]
            
            # Parse the selected matches (handle both single values and comma-separated lists)
            selected_matches = row[1]
            if ',' in selected_matches:
                # Remove quotes and split by comma
                selected_matches = selected_matches.strip('"').split(',')
                # Convert to integers
                selected_matches = [int(idx) for idx in selected_matches]
            else:
                selected_matches = [int(selected_matches)]
            
            matches[image_id] = selected_matches
    
    return matches

def load_reranked_results(file_path):
    """Load a reranked results file and return the original and reranked results"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return {
        'image_id': data['query_id'],
        'original_order': data['original_order'],
        'reranked_results': data['reranked_results']
    }

def compute_top_k_recall(correct_indices, ranking, k):
    """
    Compute Top-k recall (whether any item in the top k ranking is a correct match)
    """
    for i in range(min(k, len(ranking))):
        if ranking[i] in correct_indices:
            return 1
    return 0

def compute_ndcg(correct_indices, ranking, k=5):
    """
    Compute NDCG@k for a ranking
    NDCG = DCG / IDCG
    DCG = sum(rel_i / log2(i+1)) for i in 1 to k
    where rel_i is 1 if the item is relevant (in correct_matches) and 0 otherwise
    """
    # Calculate DCG
    dcg = 0
    for i in range(min(k, len(ranking))):
        rel = 1 if ranking[i] in correct_indices else 0
        dcg += rel / math.log2(i + 2)  # +2 because i is 0-indexed
    
    # Calculate IDCG (ideal DCG) - sort relevance scores in descending order
    relevance = [1 if i in correct_indices else 0 for i in range(5)]
    relevance.sort(reverse=True)
    
    idcg = 0
    for i, rel in enumerate(relevance[:k]):
        idcg += rel / math.log2(i + 2)
    
    # Return NDCG
    if idcg == 0:
        return 0  # Avoid division by zero
    
    return dcg / idcg

def analyze_detailed_changes(matches_file, results_dir):
    """Analyze all rankings and compute detailed changes"""
    # Load ground truth matches
    correct_matches = load_matches(matches_file)
    
    # Find all reranked result files
    result_files = glob.glob(os.path.join(results_dir, '*_reranked_*.json'))
    
    # Details storage
    improvements = []
    declines = []
    no_change = []
    
    # Process each file
    for file_path in result_files:
        try:
            result = load_reranked_results(file_path)
            image_id = result['image_id']
            
            # Skip if we don't have ground truth for this image
            if image_id not in correct_matches:
                continue
            
            # Get the correct matches for this image
            correct_indices = correct_matches[image_id]
            
            # Original ranking: positions 0, 1, 2, 3, 4
            original_ranking = [0, 1, 2, 3, 4]
            
            # Metrics for original order
            orig_top1 = compute_top_k_recall(correct_indices, original_ranking[:1], 1)
            orig_top3 = compute_top_k_recall(correct_indices, original_ranking[:3], 3)
            orig_ndcg = compute_ndcg(correct_indices, original_ranking[:5], 5)
            
            # For reranked results: extract the original positions (0-indexed)
            reranked_positions = []
            for item in result['reranked_results'][:5]:
                # Subtract 1 because positions are 1-indexed in the data but we use 0-indexed
                position = item['position'] - 1
                reranked_positions.append(position)
                
            # Metrics for reranked order
            rerank_top1 = compute_top_k_recall(correct_indices, reranked_positions[:1], 1)
            rerank_top3 = compute_top_k_recall(correct_indices, reranked_positions[:3], 3)
            rerank_ndcg = compute_ndcg(correct_indices, reranked_positions[:5], 5)

            # Determine if it's an improvement or decline
            # We'll weight Top-1 more heavily since it's most important to users
            orig_score = orig_top1 * 2 + orig_top3 + orig_ndcg
            rerank_score = rerank_top1 * 2 + rerank_top3 + rerank_ndcg
            
            item_info = {
                'image_id': image_id,
                'correct_matches': correct_indices,
                'original_ranking': original_ranking[:5],
                'reranked_positions': reranked_positions[:5],
                'original_metrics': {
                    'top1': orig_top1,
                    'top3': orig_top3,
                    'ndcg': orig_ndcg,
                    'score': orig_score
                },
                'reranked_metrics': {
                    'top1': rerank_top1,
                    'top3': rerank_top3,
                    'ndcg': rerank_ndcg,
                    'score': rerank_score
                },
                'change': rerank_score - orig_score
            }
            
            if rerank_score > orig_score:
                improvements.append(item_info)
            elif rerank_score < orig_score:
                declines.append(item_info)
            else:
                no_change.append(item_info)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Sort by absolute magnitude of change
    improvements.sort(key=lambda x: x['change'], reverse=True)
    declines.sort(key=lambda x: x['change'])
    
    return {
        'improvements': improvements,
        'declines': declines,
        'no_change': no_change
    }

def print_detailed_results(results):
    """Print detailed analysis results"""
    total = len(results['improvements']) + len(results['declines']) + len(results['no_change'])
    
    # Print summary
    print(f"\n=== DETAILED ANALYSIS SUMMARY ===")
    print(f"Total images analyzed: {total}")
    print(f"Improvements: {len(results['improvements'])} ({len(results['improvements'])/total*100:.1f}%)")
    print(f"Declines: {len(results['declines'])} ({len(results['declines'])/total*100:.1f}%)")
    print(f"No change: {len(results['no_change'])} ({len(results['no_change'])/total*100:.1f}%)")
    
    # Print top improvements
    print("\n=== TOP 5 IMPROVEMENTS ===")
    for i, item in enumerate(results['improvements'][:5]):
        print(f"{i+1}. Image ID: {item['image_id']}")
        print(f"   Correct matches: {item['correct_matches']}")
        print(f"   Original ranking: {item['original_ranking']}")
        print(f"   Reranked positions: {item['reranked_positions']}")
        print(f"   Original Top-1: {item['original_metrics']['top1']}, Top-3: {item['original_metrics']['top3']}, NDCG: {item['original_metrics']['ndcg']:.3f}")
        print(f"   Reranked Top-1: {item['reranked_metrics']['top1']}, Top-3: {item['reranked_metrics']['top3']}, NDCG: {item['reranked_metrics']['ndcg']:.3f}")
        print(f"   Score change: +{item['change']:.3f}")
        print()
    
    # Print top declines
    print("\n=== TOP 5 DECLINES ===")
    for i, item in enumerate(results['declines'][:5]):
        print(f"{i+1}. Image ID: {item['image_id']}")
        print(f"   Correct matches: {item['correct_matches']}")
        print(f"   Original ranking: {item['original_ranking']}")
        print(f"   Reranked positions: {item['reranked_positions']}")
        print(f"   Original Top-1: {item['original_metrics']['top1']}, Top-3: {item['original_metrics']['top3']}, NDCG: {item['original_metrics']['ndcg']:.3f}")
        print(f"   Reranked Top-1: {item['reranked_metrics']['top1']}, Top-3: {item['reranked_metrics']['top3']}, NDCG: {item['reranked_metrics']['ndcg']:.3f}")
        print(f"   Score change: {item['change']:.3f}")
        print()

def main():
    matches_file = "../results/matches.csv"
    results_dir = "../results"
    
    print(f"Analyzing detailed rankings for {matches_file}...")
    results = analyze_detailed_changes(matches_file, results_dir)
    
    print_detailed_results(results)

if __name__ == "__main__":
    main() 