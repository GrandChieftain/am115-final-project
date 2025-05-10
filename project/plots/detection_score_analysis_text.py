#!/usr/bin/env python3
import os
import glob
import json
import csv
import math
import pandas as pd
import numpy as np
from analyze_rankings import load_matches, load_reranked_results, compute_top_k_recall, compute_ndcg

def load_detection_scores(crops_files):
    """Load detection scores from the crops.csv files"""
    detection_scores = {}
    
    for crops_file in crops_files:
        try:
            with open(crops_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Find the column indices
                id_col = header.index('crop_url') if 'crop_url' in header else 1
                score_col = header.index('score') if 'score' in header else 4
                
                for row in reader:
                    if len(row) > max(id_col, score_col):
                        # Extract image_id from URL
                        image_url = row[id_col]
                        image_id = os.path.basename(image_url).split('.')[0]
                        
                        # Get detection score
                        try:
                            score = float(row[score_col])
                            detection_scores[image_id] = score
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            print(f"Error loading from {crops_file}: {e}")
    
    print(f"Loaded {len(detection_scores)} detection scores")
    return detection_scores

def extract_metrics_with_detection_scores(results_dir, crops_files):
    """Extract ranking metrics and detection scores for results"""
    # Load ground truth matches
    matches_file = os.path.join(results_dir, 'matches.csv')
    correct_matches = load_matches(matches_file)
    
    # Load detection scores from crops files
    detection_scores = load_detection_scores(crops_files)
    
    # Find all reranked result files
    result_files = glob.glob(os.path.join(results_dir, '*_reranked_*.json'))
    
    # Data containers
    data = []
    
    # Process each file
    for file_path in result_files:
        try:
            result = load_reranked_results(file_path)
            image_id = result['image_id']
            
            # Skip if we don't have ground truth for this image
            if image_id not in correct_matches:
                continue
            
            # Skip if we don't have detection score for this image
            if image_id not in detection_scores:
                continue
            
            # Get correct matches and detection score
            correct_indices = correct_matches[image_id]
            detection_score = detection_scores[image_id]
            
            # Calculate metrics
            # Original ranking: positions 0, 1, 2, 3, 4
            original_ranking = [0, 1, 2, 3, 4]
            
            # Calculate metrics for original ranking
            orig_top1 = compute_top_k_recall(correct_indices, original_ranking[:1], 1)
            orig_top2 = compute_top_k_recall(correct_indices, original_ranking[:2], 2)
            orig_top3 = compute_top_k_recall(correct_indices, original_ranking[:3], 3)
            orig_ndcg = compute_ndcg(correct_indices, original_ranking[:5], 5)
            
            # Extract reranked positions
            reranked_positions = []
            for item in result['reranked_results'][:5]:
                # Convert to 0-indexed
                position = item['position'] - 1
                reranked_positions.append(position)
            
            # Calculate metrics for reranked positions
            rerank_top1 = compute_top_k_recall(correct_indices, reranked_positions[:1], 1)
            rerank_top2 = compute_top_k_recall(correct_indices, reranked_positions[:2], 2)
            rerank_top3 = compute_top_k_recall(correct_indices, reranked_positions[:3], 3)
            rerank_ndcg = compute_ndcg(correct_indices, reranked_positions[:5], 5)
            
            # Store all data
            data.append({
                'image_id': image_id,
                'detection_score': detection_score,
                'original_top1': orig_top1,
                'original_top2': orig_top2,
                'original_top3': orig_top3,
                'original_ndcg': orig_ndcg,
                'reranked_top1': rerank_top1,
                'reranked_top2': rerank_top2,
                'reranked_top3': rerank_top3,
                'reranked_ndcg': rerank_ndcg,
                'num_correct_matches': len(correct_indices)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(data)

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient without scipy"""
    n = len(x)
    if n == 0:
        return 0.0, 1.0
    
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
    
    # We're skipping p-value calculation as it's complex
    # Just return a placeholder
    p_value = 0.0
    
    return correlation, p_value

def analyze_correlation(df):
    """Analyze correlation between detection score and ranking metrics"""
    correlations = {}
    
    # Calculate Pearson correlation for each metric
    for metric in ['original_top1', 'original_top2', 'original_top3', 'original_ndcg',
                  'reranked_top1', 'reranked_top2', 'reranked_top3', 'reranked_ndcg']:
        correlation, _ = calculate_correlation(df['detection_score'].tolist(), df[metric].tolist())
        correlations[metric] = {
            'correlation': correlation,
            'p_value': 0.0  # Placeholder
        }
    
    # Sort by correlation strength
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    
    # Print results
    print("\n=== CORRELATION ANALYSIS ===")
    print("Pearson correlation between detection score and ranking metrics:")
    for metric, stats_data in sorted_correlations:
        correlation = stats_data['correlation']
        print(f"  {metric}: r = {correlation:.4f}")
    
    return correlations

def bin_by_detection_score(df, num_bins=5):
    """Bin images by detection score and calculate average metrics for each bin"""
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Sort by detection_score
    df_copy = df_copy.sort_values('detection_score')
    
    # Calculate bin size
    bin_size = len(df_copy) // num_bins
    remainder = len(df_copy) % num_bins
    
    # Create bins with approximately equal number of samples
    bins = []
    start_idx = 0
    
    for i in range(num_bins):
        # Add one extra item to early bins if we have a remainder
        extra = 1 if i < remainder else 0
        end_idx = start_idx + bin_size + extra
        
        # Get subset for this bin
        bin_df = df_copy.iloc[start_idx:end_idx]
        
        # Calculate metrics for bin
        bin_metrics = {
            'bin_index': i,
            'detection_score': bin_df['detection_score'].mean(),
            'detection_score_min': bin_df['detection_score'].min(),
            'detection_score_max': bin_df['detection_score'].max(),
            'original_top1': bin_df['original_top1'].mean(),
            'original_top2': bin_df['original_top2'].mean(),
            'original_top3': bin_df['original_top3'].mean(),
            'original_ndcg': bin_df['original_ndcg'].mean(),
            'reranked_top1': bin_df['reranked_top1'].mean(),
            'reranked_top2': bin_df['reranked_top2'].mean(),
            'reranked_top3': bin_df['reranked_top3'].mean(),
            'reranked_ndcg': bin_df['reranked_ndcg'].mean(),
            'count': len(bin_df)
        }
        
        bins.append(bin_metrics)
        start_idx = end_idx
    
    return pd.DataFrame(bins)

def export_results_to_csv(df, bin_metrics, output_dir='.'):
    """Export results to CSV files"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export raw data
    df.to_csv(os.path.join(output_dir, 'detection_score_metrics.csv'), index=False)
    
    # Export binned data
    bin_metrics.to_csv(os.path.join(output_dir, 'detection_score_bins.csv'), index=False)
    
    # Calculate improvement percentages for binned data
    bin_metrics['top1_improvement'] = (bin_metrics['reranked_top1'] - bin_metrics['original_top1']) / bin_metrics['original_top1'] * 100
    bin_metrics.loc[bin_metrics['original_top1'] == 0, 'top1_improvement'] = 0  # Handle division by zero
    
    bin_metrics['top2_improvement'] = (bin_metrics['reranked_top2'] - bin_metrics['original_top2']) / bin_metrics['original_top2'] * 100
    bin_metrics.loc[bin_metrics['original_top2'] == 0, 'top2_improvement'] = 0  # Handle division by zero
    
    bin_metrics['top3_improvement'] = (bin_metrics['reranked_top3'] - bin_metrics['original_top3']) / bin_metrics['original_top3'] * 100
    bin_metrics.loc[bin_metrics['original_top3'] == 0, 'top3_improvement'] = 0  # Handle division by zero
    
    bin_metrics['ndcg_improvement'] = (bin_metrics['reranked_ndcg'] - bin_metrics['original_ndcg']) / bin_metrics['original_ndcg'] * 100
    bin_metrics.loc[bin_metrics['original_ndcg'] == 0, 'ndcg_improvement'] = 0  # Handle division by zero
    
    # Export improvement data
    bin_metrics[['bin_index', 'detection_score', 'count', 
                'top1_improvement', 'top2_improvement', 'top3_improvement', 'ndcg_improvement']].to_csv(
        os.path.join(output_dir, 'detection_score_improvement.csv'), index=False)
    
    return

def main():
    results_dir = "../results"
    output_dir = "."
    
    # Use both men's and women's outfit crop files
    crops_files = [
        "../data/mens_outfits.crops.csv",
        "../data/women_outfits.crops.csv"
    ]
    
    print("Extracting metrics and detection scores...")
    df = extract_metrics_with_detection_scores(results_dir, crops_files)
    
    # Check if we have any valid data
    if df.empty:
        print("No valid detection scores found for images with matches.")
        return
    
    print(f"Found {len(df)} images with both valid detection scores and match data")
    
    # Analyze correlation
    correlations = analyze_correlation(df)
    
    # Bin by detection score
    print("\nBinning images by detection score...")
    bin_metrics = bin_by_detection_score(df)
    
    # Export results to CSV
    print("Exporting results to CSV...")
    export_results_to_csv(df, bin_metrics, output_dir)
    
    # Print summary of binned data
    print("\n=== DETECTION SCORE BIN ANALYSIS ===")
    for i, row in bin_metrics.iterrows():
        bin_index = row['bin_index']
        detection_score = row['detection_score']
        count = row['count']
        
        print(f"\nBin {bin_index}: Avg Detection Score = {detection_score:.4f} (n={count})")
        print(f"  Original Top-1: {row['original_top1']:.4f}, Reranked: {row['reranked_top1']:.4f}, Change: {row['top1_improvement']:.1f}%")
        print(f"  Original Top-2: {row['original_top2']:.4f}, Reranked: {row['reranked_top2']:.4f}, Change: {row['top2_improvement']:.1f}%")
        print(f"  Original Top-3: {row['original_top3']:.4f}, Reranked: {row['reranked_top3']:.4f}, Change: {row['top3_improvement']:.1f}%")
        print(f"  Original NDCG@5: {row['original_ndcg']:.4f}, Reranked: {row['reranked_ndcg']:.4f}, Change: {row['ndcg_improvement']:.1f}%")
    
    print("\nAnalysis complete. Output files have been saved to the output directory.")

if __name__ == "__main__":
    main() 