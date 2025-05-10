#!/usr/bin/env python3
import os
import glob
import json
import csv
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from analyze_rankings import load_matches, load_reranked_results, compute_top_k_recall, compute_ndcg

def load_categories_and_scores(crops_files):
    """Load categories and detection scores from the crops.csv files"""
    categories = {}
    detection_scores = {}
    
    for crops_file in crops_files:
        try:
            with open(crops_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Find the column indices
                id_col = header.index('crop_url') if 'crop_url' in header else 1
                label_col = header.index('label') if 'label' in header else 3
                score_col = header.index('score') if 'score' in header else 4
                
                for row in reader:
                    if len(row) > max(id_col, label_col, score_col):
                        # Extract image_id from URL
                        image_url = row[id_col]
                        image_id = os.path.basename(image_url).split('.')[0]
                        
                        # Get category and detection score
                        category = row[label_col]
                        try:
                            score = float(row[score_col])
                        except (ValueError, TypeError):
                            score = 0.0
                        
                        # Store data
                        categories[image_id] = category
                        detection_scores[image_id] = score
        except Exception as e:
            print(f"Error loading from {crops_file}: {e}")
        
    print(f"Loaded {len(categories)} categories and {len(detection_scores)} detection scores")
    return categories, detection_scores

def calculate_kendall_tau(original_positions, reranked_positions):
    """Calculate Kendall Tau distance between original and reranked positions"""
    # Convert positions to rankings (position 0 is rank 1, etc.)
    original_ranks = list(range(1, len(original_positions) + 1))
    
    # For reranked, we need to get the ranking of the original positions
    reranked_ranks = []
    for pos in original_positions:
        try:
            rank = reranked_positions.index(pos) + 1
        except ValueError:
            # If position not in reranked, give it a low rank
            rank = len(reranked_positions) + 1
        reranked_ranks.append(rank)
    
    # Calculate Kendall Tau without using scipy
    # This is a simplified version that calculates the correlation
    n = len(original_ranks)
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if (original_ranks[i] - original_ranks[j]) * (reranked_ranks[i] - reranked_ranks[j]) > 0:
                concordant += 1
            elif (original_ranks[i] - original_ranks[j]) * (reranked_ranks[i] - reranked_ranks[j]) < 0:
                discordant += 1
    
    total_pairs = n * (n - 1) // 2
    tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
    
    return tau, 0.0  # Return 0.0 as p-value placeholder

def analyze_by_category(matches_file, results_dir, crops_files):
    """Analyze rankings by category"""
    # Load ground truth matches
    correct_matches = load_matches(matches_file)
    
    # Load categories and detection scores
    categories, detection_scores = load_categories_and_scores(crops_files)
    
    # Find all reranked result files
    result_files = glob.glob(os.path.join(results_dir, '*_reranked_*.json'))
    
    # Initialize category metrics
    category_metrics = defaultdict(lambda: {
        'count': 0,
        'original': {'top1': 0, 'top2': 0, 'top3': 0, 'ndcg5': 0},
        'reranked': {'top1': 0, 'top2': 0, 'top3': 0, 'ndcg5': 0},
        'kendall_tau': [],
        'detection_scores': [],
        'original_accuracy': [],
        'reranked_accuracy': []
    })
    
    # Process each file
    total_processed = 0
    with_category = 0
    
    for file_path in result_files:
        try:
            result = load_reranked_results(file_path)
            image_id = result['image_id']
            
            # Skip if we don't have ground truth for this image
            if image_id not in correct_matches:
                continue
            
            total_processed += 1
            
            # Get category if available, otherwise use "unknown"
            category = categories.get(image_id, "unknown")
            if category != "unknown":
                with_category += 1
            
            # Get detection score if available
            detection_score = detection_scores.get(image_id, 0.0)
            
            # Get correct matches
            correct_indices = correct_matches[image_id]
            
            # Original ranking
            original_ranking = [0, 1, 2, 3, 4]
            
            # Calculate metrics for original ranking
            orig_top1 = compute_top_k_recall(correct_indices, original_ranking[:1], 1)
            orig_top2 = compute_top_k_recall(correct_indices, original_ranking[:2], 2)
            orig_top3 = compute_top_k_recall(correct_indices, original_ranking[:3], 3)
            orig_ndcg = compute_ndcg(correct_indices, original_ranking[:5], 5)
            
            # Extract reranked positions
            reranked_positions = []
            for item in result['reranked_results'][:5]:
                position = item['position'] - 1  # Convert to 0-indexed
                reranked_positions.append(position)
            
            # Calculate metrics for reranked positions
            rerank_top1 = compute_top_k_recall(correct_indices, reranked_positions[:1], 1)
            rerank_top2 = compute_top_k_recall(correct_indices, reranked_positions[:2], 2)
            rerank_top3 = compute_top_k_recall(correct_indices, reranked_positions[:3], 3)
            rerank_ndcg = compute_ndcg(correct_indices, reranked_positions[:5], 5)
            
            # Calculate Kendall Tau
            tau, _ = calculate_kendall_tau(original_ranking, reranked_positions)
            
            # Update category metrics
            category_metrics[category]['count'] += 1
            category_metrics[category]['original']['top1'] += orig_top1
            category_metrics[category]['original']['top2'] += orig_top2
            category_metrics[category]['original']['top3'] += orig_top3
            category_metrics[category]['original']['ndcg5'] += orig_ndcg
            category_metrics[category]['reranked']['top1'] += rerank_top1
            category_metrics[category]['reranked']['top2'] += rerank_top2
            category_metrics[category]['reranked']['top3'] += rerank_top3
            category_metrics[category]['reranked']['ndcg5'] += rerank_ndcg
            category_metrics[category]['kendall_tau'].append(tau)
            
            # Store detection score and accuracy for regression analysis
            category_metrics[category]['detection_scores'].append(detection_score)
            category_metrics[category]['original_accuracy'].append(orig_top1)
            category_metrics[category]['reranked_accuracy'].append(rerank_top1)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Processed {total_processed} images, {with_category} with categories")
    
    # Calculate averages for each category
    for category, metrics in category_metrics.items():
        if metrics['count'] > 0:
            metrics['original']['top1'] /= metrics['count']
            metrics['original']['top3'] /= metrics['count']
            metrics['original']['ndcg5'] /= metrics['count']
            metrics['reranked']['top1'] /= metrics['count']
            metrics['reranked']['top3'] /= metrics['count']
            metrics['reranked']['ndcg5'] /= metrics['count']
            metrics['avg_kendall_tau'] = sum(metrics['kendall_tau']) / len(metrics['kendall_tau'])
    
    return category_metrics

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient without scipy"""
    n = len(x)
    if n == 0:
        return 0.0
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate covariance and variances
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    variance_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    variance_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    # Calculate correlation
    if variance_x > 0 and variance_y > 0:
        correlation = covariance / (math.sqrt(variance_x) * math.sqrt(variance_y))
    else:
        correlation = 0.0
    
    return correlation

def export_metrics_to_csv(category_metrics, output_dir='.'):
    """Export category metrics to CSV without matplotlib"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for output
    categories = []
    counts = []
    orig_top1 = []
    orig_top2 = []
    orig_top3 = []
    orig_ndcg = []
    rerank_top1 = []
    rerank_top2 = []
    rerank_top3 = []
    rerank_ndcg = []
    kendall_taus = []
    
    for category, metrics in category_metrics.items():
        if metrics['count'] > 0:
            categories.append(category)
            counts.append(metrics['count'])
            orig_top1.append(metrics['original']['top1'])
            orig_top2.append(metrics['original']['top2'])
            orig_top3.append(metrics['original']['top3'])
            orig_ndcg.append(metrics['original']['ndcg5'])
            rerank_top1.append(metrics['reranked']['top1'])
            rerank_top2.append(metrics['reranked']['top2'])
            rerank_top3.append(metrics['reranked']['top3'])
            rerank_ndcg.append(metrics['reranked']['ndcg5'])
            kendall_taus.append(metrics['avg_kendall_tau'])
    
    # Sort categories by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    categories = [categories[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    orig_top1 = [orig_top1[i] for i in sorted_indices]
    orig_top2 = [orig_top2[i] for i in sorted_indices]
    orig_top3 = [orig_top3[i] for i in sorted_indices]
    orig_ndcg = [orig_ndcg[i] for i in sorted_indices]
    rerank_top1 = [rerank_top1[i] for i in sorted_indices]
    rerank_top2 = [rerank_top2[i] for i in sorted_indices]
    rerank_top3 = [rerank_top3[i] for i in sorted_indices]
    rerank_ndcg = [rerank_ndcg[i] for i in sorted_indices]
    kendall_taus = [kendall_taus[i] for i in sorted_indices]
    
    # Create a table of results
    result_df = pd.DataFrame({
        'Category': categories,
        'Count': counts,
        'Original Top-1': orig_top1,
        'Reranked Top-1': rerank_top1,
        'Original Top-2': orig_top2,
        'Reranked Top-2': rerank_top2,
        'Original Top-3': orig_top3,
        'Reranked Top-3': rerank_top3,
        'Original NDCG@5': orig_ndcg,
        'Reranked NDCG@5': rerank_ndcg,
        'Kendall Tau': kendall_taus
    })
    
    # Calculate improvement percentages
    result_df['Top-1 Improvement'] = (result_df['Reranked Top-1'] - result_df['Original Top-1']) / result_df['Original Top-1'] * 100
    result_df.loc[result_df['Original Top-1'] == 0, 'Top-1 Improvement'] = 0  # Handle division by zero

    result_df['Top-2 Improvement'] = (result_df['Reranked Top-2'] - result_df['Original Top-2']) / result_df['Original Top-2'] * 100
    result_df.loc[result_df['Original Top-2'] == 0, 'Top-2 Improvement'] = 0  # Handle division by zero
    
    result_df['Top-3 Improvement'] = (result_df['Reranked Top-3'] - result_df['Original Top-3']) / result_df['Original Top-3'] * 100
    result_df.loc[result_df['Original Top-3'] == 0, 'Top-3 Improvement'] = 0  # Handle division by zero
    
    result_df['NDCG@5 Improvement'] = (result_df['Reranked NDCG@5'] - result_df['Original NDCG@5']) / result_df['Original NDCG@5'] * 100
    result_df.loc[result_df['Original NDCG@5'] == 0, 'NDCG@5 Improvement'] = 0  # Handle division by zero
    
    # Save to CSV
    result_df.to_csv(os.path.join(output_dir, 'category_metrics.csv'), index=False)
    
    # Calculate correlation between detection score and metrics
    for category in categories:
        metrics = category_metrics[category]
        if len(metrics['detection_scores']) > 0:
            corr_orig = calculate_correlation(metrics['detection_scores'], metrics['original_accuracy'])
            corr_rerank = calculate_correlation(metrics['detection_scores'], metrics['reranked_accuracy'])
            
            print(f"\nCorrelation for category '{category}':")
            print(f"  Detection Score vs. Original Top-1: {corr_orig:.4f}")
            print(f"  Detection Score vs. Reranked Top-1: {corr_rerank:.4f}")
    
    return result_df

def main():
    matches_file = "../results/matches.csv"
    results_dir = "../results"
    
    # Use both men's and women's outfit crop files
    crops_files = [
        "../data/mens_outfits.crops.csv",
        "../data/women_outfits.crops.csv"
    ]
    
    output_dir = "."  # Current directory for output
    
    print(f"Analyzing rankings by category...")
    category_metrics = analyze_by_category(matches_file, results_dir, crops_files)
    
    print(f"\nExporting metrics to CSV...")
    result_df = export_metrics_to_csv(category_metrics, output_dir)
    
    # Print summary
    print("\n=== CATEGORY ANALYSIS SUMMARY ===")
    
    for category in result_df['Category']:
        row = result_df[result_df['Category'] == category].iloc[0]
        print(f"\nCategory: {category} (n={row['Count']})")
        print(f"  Original Top-1: {row['Original Top-1']:.4f}, Reranked: {row['Reranked Top-1']:.4f}, Change: {row['Top-1 Improvement']:.1f}%")
        print(f"  Original Top-2: {row['Original Top-2']:.4f}, Reranked: {row['Reranked Top-2']:.4f}, Change: {row['Top-2 Improvement']:.1f}%")
        print(f"  Original Top-3: {row['Original Top-3']:.4f}, Reranked: {row['Reranked Top-3']:.4f}, Change: {row['Top-3 Improvement']:.1f}%")
        print(f"  Original NDCG@5: {row['Original NDCG@5']:.4f}, Reranked: {row['Reranked NDCG@5']:.4f}, Change: {row['NDCG@5 Improvement']:.1f}%")
        print(f"  Kendall Tau: {row['Kendall Tau']:.4f}")
    
if __name__ == "__main__":
    main() 