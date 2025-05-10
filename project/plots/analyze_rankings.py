#!/usr/bin/env python3
import os
import json
import csv
import math
import glob
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

def analyze_rankings(matches_file, results_dir):
    """Analyze all rankings and compute metrics"""
    # Load ground truth matches
    correct_matches = load_matches(matches_file)
    
    # Find all reranked result files
    result_files = glob.glob(os.path.join(results_dir, '*_reranked_*.json'))
    
    # Metrics storage
    metrics = {
        'original': {
            'top1': 0,
            'top2': 0,
            'top3': 0,
            'ndcg5': 0,
        },
        'reranked': {
            'top1': 0,
            'top2': 0,
            'top3': 0,
            'ndcg5': 0,
        }
    }
    
    # Track how many images we process
    total_processed = 0
    
    # Process each file
    for file_path in result_files:
        try:
            result = load_reranked_results(file_path)
            image_id = result['image_id']
            
            # Skip if we don't have ground truth for this image
            if image_id not in correct_matches:
                print(f"Skipping {image_id} - no ground truth found")
                continue
            
            # Get the correct matches for this image
            correct_indices = correct_matches[image_id]
            
            # Original ranking: positions 0, 1, 2, 3, 4 
            # (we're checking whether these indices are in our correct_indices)
            original_ranking = [0, 1, 2, 3, 4]
            
            metrics['original']['top1'] += compute_top_k_recall(correct_indices, original_ranking[:1], 1)
            metrics['original']['top2'] += compute_top_k_recall(correct_indices, original_ranking[:2], 2)
            metrics['original']['top3'] += compute_top_k_recall(correct_indices, original_ranking[:3], 3)
            metrics['original']['ndcg5'] += compute_ndcg(correct_indices, original_ranking[:5], 5)
            
            # For reranked results: extract the original positions (0-indexed)
            # The position in reranked_results are the original 1-indexed positions
            # We need to convert them to 0-indexed for our evaluation
            reranked_positions = []
            for item in result['reranked_results'][:5]:
                # Subtract 1 because positions are 1-indexed in the data but we use 0-indexed
                position = item['position'] - 1
                reranked_positions.append(position)
                
            metrics['reranked']['top1'] += compute_top_k_recall(correct_indices, reranked_positions[:1], 1)
            metrics['reranked']['top2'] += compute_top_k_recall(correct_indices, reranked_positions[:2], 2)
            metrics['reranked']['top3'] += compute_top_k_recall(correct_indices, reranked_positions[:3], 3)
            metrics['reranked']['ndcg5'] += compute_ndcg(correct_indices, reranked_positions[:5], 5)
            
            total_processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Compute averages
    if total_processed > 0:
        for ranking_type in metrics:
            for metric in metrics[ranking_type]:
                metrics[ranking_type][metric] /= total_processed
    
    return metrics, total_processed

def main():
    matches_file = "../results/matches.csv"
    results_dir = "../results"
    
    print(f"Analyzing rankings for {matches_file}...")
    metrics, total_processed = analyze_rankings(matches_file, results_dir)
    
    print(f"\nProcessed {total_processed} images\n")
    
    print("Original Google Lens Rankings:")
    print(f"  Top-1 Accuracy: {metrics['original']['top1']:.4f}")
    print(f"  Top-2 Accuracy: {metrics['original']['top2']:.4f}")
    print(f"  Top-3 Accuracy: {metrics['original']['top3']:.4f}")
    print(f"  NDCG@5: {metrics['original']['ndcg5']:.4f}")
    
    print("\nReranked Results:")
    print(f"  Top-1 Accuracy: {metrics['reranked']['top1']:.4f}")
    print(f"  Top-2 Accuracy: {metrics['reranked']['top2']:.4f}")
    print(f"  Top-3 Accuracy: {metrics['reranked']['top3']:.4f}")
    print(f"  NDCG@5: {metrics['reranked']['ndcg5']:.4f}")
    
    # Calculate improvement percentages
    print("\nImprovement:")
    for metric in ['top1', 'top2', 'top3', 'ndcg5']:
        orig = metrics['original'][metric]
        rerank = metrics['reranked'][metric]
        rel_improvement = ((rerank - orig) / orig) * 100 if orig > 0 else float('inf')
        abs_improvement = rerank - orig
        print(f"  {metric}: {rel_improvement:.2f}% (absolute: {abs_improvement:.4f})")

if __name__ == "__main__":
    main() 