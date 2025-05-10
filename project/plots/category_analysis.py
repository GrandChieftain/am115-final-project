#!/usr/bin/env python3
import os
import glob
import json
import csv
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
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
    
    # Calculate Kendall Tau
    tau, p_value = kendalltau(original_ranks, reranked_ranks)
    
    return tau, p_value

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
        'original': {'top1': 0, 'top3': 0, 'ndcg5': 0},
        'reranked': {'top1': 0, 'top3': 0, 'ndcg5': 0},
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
            orig_top3 = compute_top_k_recall(correct_indices, original_ranking[:3], 3)
            orig_ndcg = compute_ndcg(correct_indices, original_ranking[:5], 5)
            
            # Extract reranked positions
            reranked_positions = []
            for item in result['reranked_results'][:5]:
                position = item['position'] - 1  # Convert to 0-indexed
                reranked_positions.append(position)
            
            # Calculate metrics for reranked positions
            rerank_top1 = compute_top_k_recall(correct_indices, reranked_positions[:1], 1)
            rerank_top3 = compute_top_k_recall(correct_indices, reranked_positions[:3], 3)
            rerank_ndcg = compute_ndcg(correct_indices, reranked_positions[:5], 5)
            
            # Calculate Kendall Tau
            tau, _ = calculate_kendall_tau(original_ranking, reranked_positions)
            
            # Update category metrics
            category_metrics[category]['count'] += 1
            category_metrics[category]['original']['top1'] += orig_top1
            category_metrics[category]['original']['top3'] += orig_top3
            category_metrics[category]['original']['ndcg5'] += orig_ndcg
            category_metrics[category]['reranked']['top1'] += rerank_top1
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

def visualize_category_metrics(category_metrics, output_dir='.'):
    """Create visualizations for category metrics"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    categories = []
    counts = []
    orig_top1 = []
    orig_top3 = []
    orig_ndcg = []
    rerank_top1 = []
    rerank_top3 = []
    rerank_ndcg = []
    kendall_taus = []
    
    for category, metrics in category_metrics.items():
        if metrics['count'] > 0:
            categories.append(category)
            counts.append(metrics['count'])
            orig_top1.append(metrics['original']['top1'])
            orig_top3.append(metrics['original']['top3'])
            orig_ndcg.append(metrics['original']['ndcg5'])
            rerank_top1.append(metrics['reranked']['top1'])
            rerank_top3.append(metrics['reranked']['top3'])
            rerank_ndcg.append(metrics['reranked']['ndcg5'])
            kendall_taus.append(metrics['avg_kendall_tau'])
    
    # Sort categories by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    categories = [categories[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    orig_top1 = [orig_top1[i] for i in sorted_indices]
    orig_top3 = [orig_top3[i] for i in sorted_indices]
    orig_ndcg = [orig_ndcg[i] for i in sorted_indices]
    rerank_top1 = [rerank_top1[i] for i in sorted_indices]
    rerank_top3 = [rerank_top3[i] for i in sorted_indices]
    rerank_ndcg = [rerank_ndcg[i] for i in sorted_indices]
    kendall_taus = [kendall_taus[i] for i in sorted_indices]
    
    # 1. Create bar chart for Top-1 by category
    plt.figure(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, orig_top1, width, label='Original Top-1')
    plt.bar(x + width/2, rerank_top1, width, label='Reranked Top-1')
    
    plt.xlabel('Category')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Top-1 Accuracy by Category')
    plt.xticks(x, categories, rotation=45, ha='right')
    
    # Add count as text above each category
    for i, count in enumerate(counts):
        plt.text(i, 0.05, f"n={count}", ha='center', fontsize=8)
    
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_top1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create bar chart for NDCG@5 by category
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width/2, orig_ndcg, width, label='Original NDCG@5')
    plt.bar(x + width/2, rerank_ndcg, width, label='Reranked NDCG@5')
    
    plt.xlabel('Category')
    plt.ylabel('NDCG@5')
    plt.title('NDCG@5 by Category')
    plt.xticks(x, categories, rotation=45, ha='right')
    
    # Add count as text above each category
    for i, count in enumerate(counts):
        plt.text(i, 0.05, f"n={count}", ha='center', fontsize=8)
    
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_ndcg.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create bar chart for Kendall Tau by category
    plt.figure(figsize=(12, 6))
    
    plt.bar(x, kendall_taus, width)
    
    plt.xlabel('Category')
    plt.ylabel('Average Kendall Tau')
    plt.title('Average Kendall Tau by Category')
    plt.xticks(x, categories, rotation=45, ha='right')
    
    # Add count as text above each category
    for i, count in enumerate(counts):
        plt.text(i, 0.05, f"n={count}", ha='center', fontsize=8)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_kendall_tau.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create scatter plot for detection score vs. accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data for regression analysis
    all_detection_scores = []
    all_original_accuracies = []
    all_reranked_accuracies = []
    
    for metrics in category_metrics.values():
        all_detection_scores.extend(metrics['detection_scores'])
        all_original_accuracies.extend(metrics['original_accuracy'])
        all_reranked_accuracies.extend(metrics['reranked_accuracy'])
    
    # Plot original accuracy vs detection score
    ax1.scatter(all_detection_scores, all_original_accuracies, alpha=0.5)
    ax1.set_xlabel('Detection Score')
    ax1.set_ylabel('Top-1 Accuracy (Original)')
    ax1.set_title('Detection Score vs. Original Ranking Accuracy')
    
    # Add regression line for original
    if all_detection_scores and any(ds > 0 for ds in all_detection_scores):
        z = np.polyfit(all_detection_scores, all_original_accuracies, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(all_detection_scores), max(all_detection_scores), 100)
        ax1.plot(x_range, p(x_range), "r--")
        
        # Calculate correlation
        corr = np.corrcoef(all_detection_scores, all_original_accuracies)[0, 1]
        ax1.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top')
    
    # Plot reranked accuracy vs detection score
    ax2.scatter(all_detection_scores, all_reranked_accuracies, alpha=0.5)
    ax2.set_xlabel('Detection Score')
    ax2.set_ylabel('Top-1 Accuracy (Reranked)')
    ax2.set_title('Detection Score vs. Reranked Ranking Accuracy')
    
    # Add regression line for reranked
    if all_detection_scores and any(ds > 0 for ds in all_detection_scores):
        z = np.polyfit(all_detection_scores, all_reranked_accuracies, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(all_detection_scores), max(all_detection_scores), 100)
        ax2.plot(x_range, p(x_range), "r--")
        
        # Calculate correlation
        corr = np.corrcoef(all_detection_scores, all_reranked_accuracies)[0, 1]
        ax2.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_score_regression.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create a table of results
    result_df = pd.DataFrame({
        'Category': categories,
        'Count': counts,
        'Original Top-1': orig_top1,
        'Reranked Top-1': rerank_top1,
        'Original Top-3': orig_top3,
        'Reranked Top-3': rerank_top3,
        'Original NDCG@5': orig_ndcg,
        'Reranked NDCG@5': rerank_ndcg,
        'Kendall Tau': kendall_taus
    })
    
    # Calculate improvement percentages
    result_df['Top-1 Improvement'] = (result_df['Reranked Top-1'] - result_df['Original Top-1']) / result_df['Original Top-1'] * 100
    result_df.loc[result_df['Original Top-1'] == 0, 'Top-1 Improvement'] = 0  # Handle division by zero
    
    result_df['Top-3 Improvement'] = (result_df['Reranked Top-3'] - result_df['Original Top-3']) / result_df['Original Top-3'] * 100
    result_df.loc[result_df['Original Top-3'] == 0, 'Top-3 Improvement'] = 0  # Handle division by zero
    
    result_df['NDCG@5 Improvement'] = (result_df['Reranked NDCG@5'] - result_df['Original NDCG@5']) / result_df['Original NDCG@5'] * 100
    result_df.loc[result_df['Original NDCG@5'] == 0, 'NDCG@5 Improvement'] = 0  # Handle division by zero
    
    # Save to CSV
    result_df.to_csv(os.path.join(output_dir, 'category_metrics.csv'), index=False)
    
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
    
    print(f"Creating visualizations...")
    result_df = visualize_category_metrics(category_metrics, output_dir)
    
    # Print summary
    print("\n=== CATEGORY ANALYSIS SUMMARY ===")
    
    for category in result_df['Category']:
        row = result_df[result_df['Category'] == category].iloc[0]
        print(f"\nCategory: {category} (n={row['Count']})")
        print(f"  Original Top-1: {row['Original Top-1']:.4f}, Reranked: {row['Reranked Top-1']:.4f}, Change: {row['Top-1 Improvement']:.1f}%")
        print(f"  Original Top-3: {row['Original Top-3']:.4f}, Reranked: {row['Reranked Top-3']:.4f}, Change: {row['Top-3 Improvement']:.1f}%")
        print(f"  Original NDCG@5: {row['Original NDCG@5']:.4f}, Reranked: {row['Reranked NDCG@5']:.4f}, Change: {row['NDCG@5 Improvement']:.1f}%")
        print(f"  Kendall Tau: {row['Kendall Tau']:.4f}")
    
if __name__ == "__main__":
    main() 