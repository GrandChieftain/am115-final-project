#!/usr/bin/env python3
import os
import glob
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from analyze_rankings import load_matches, load_reranked_results, compute_top_k_recall, compute_ndcg
from sklearn.linear_model import LinearRegression

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

def analyze_correlation(df):
    """Analyze correlation between detection score and ranking metrics"""
    correlations = {}
    
    # Calculate Pearson correlation for each metric
    for metric in ['original_top1', 'original_top2', 'original_top3', 'original_ndcg',
                  'reranked_top1', 'reranked_top2', 'reranked_top3', 'reranked_ndcg']:
        correlation, p_value = stats.pearsonr(df['detection_score'], df[metric])
        correlations[metric] = {
            'correlation': correlation,
            'p_value': p_value
        }
    
    # Sort by correlation strength
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    
    # Print results
    print("\n=== CORRELATION ANALYSIS ===")
    print("Pearson correlation between detection score and ranking metrics:")
    for metric, stats_data in sorted_correlations:
        correlation = stats_data['correlation']
        p_value = stats_data['p_value']
        significance = "*** (p<0.001)" if p_value < 0.001 else "** (p<0.01)" if p_value < 0.01 else "* (p<0.05)" if p_value < 0.05 else "(not significant)"
        print(f"  {metric}: r = {correlation:.4f}, p = {p_value:.4f} {significance}")
    
    return correlations

def bin_by_detection_score(df, num_bins=5):
    """Bin images by detection score and calculate average metrics for each bin"""
    # Create bins
    df['detection_bin'] = pd.qcut(df['detection_score'], num_bins, labels=False)
    
    # Calculate metrics for each bin
    bin_metrics = df.groupby('detection_bin').agg({
        'detection_score': 'mean',
        'original_top1': 'mean',
        'original_top2': 'mean',
        'original_top3': 'mean',
        'original_ndcg': 'mean',
        'reranked_top1': 'mean',
        'reranked_top2': 'mean',
        'reranked_top3': 'mean',
        'reranked_ndcg': 'mean',
        'image_id': 'count'
    }).reset_index()
    
    # Rename count column
    bin_metrics = bin_metrics.rename(columns={'image_id': 'count'})
    
    return bin_metrics

def create_visualizations(df, bin_metrics, output_dir='.'):
    """Create visualizations for detection score correlation analysis"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plots with regression lines
    metrics = [
        ('original_top1', 'Original Top-1'),
        ('reranked_top1', 'Reranked Top-1'),
        ('original_ndcg', 'Original NDCG@5'),
        ('reranked_ndcg', 'Reranked NDCG@5')
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    for i, (metric, label) in enumerate(metrics):
        ax = axs[i]
        
        # Scatter plot
        ax.scatter(df['detection_score'], df[metric], alpha=0.5)
        
        # Fit regression line
        X = df['detection_score'].values.reshape(-1, 1)
        y = df[metric].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Add regression line
        x_range = np.linspace(df['detection_score'].min(), df['detection_score'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, 'r--')
        
        # Calculate and display R²
        r_squared = model.score(X, y)
        correlation, p_value = stats.pearsonr(df['detection_score'], df[metric])
        ax.text(0.05, 0.95, f"R² = {r_squared:.4f}\nr = {correlation:.4f}\np = {p_value:.4f}", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        # Set labels and title
        ax.set_xlabel('Detection Score')
        ax.set_ylabel(label)
        ax.set_title(f'Detection Score vs. {label}')
        
        # Set y-axis limits for accuracy metrics
        if 'top' in metric:
            ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_score_regression_all.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Binned analysis plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top-1 by bin
    width = 0.35
    x = np.arange(len(bin_metrics))
    
    # Plot for Top-1
    axs[0].bar(x - width/2, bin_metrics['original_top1'], width, label='Original')
    axs[0].bar(x + width/2, bin_metrics['reranked_top1'], width, label='Reranked')
    
    # Set x-ticks to show detection score range for each bin
    x_labels = [f"{bin_metrics['detection_score'][i]:.2f}\n(n={bin_metrics['count'][i]})" for i in x]
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(x_labels)
    
    axs[0].set_xlabel('Average Detection Score in Bin (and count)')
    axs[0].set_ylabel('Top-1 Accuracy')
    axs[0].set_title('Top-1 Accuracy by Detection Score Bin')
    axs[0].set_ylim(0, 1.0)
    axs[0].legend()
    
    # Plot for NDCG
    axs[1].bar(x - width/2, bin_metrics['original_ndcg'], width, label='Original')
    axs[1].bar(x + width/2, bin_metrics['reranked_ndcg'], width, label='Reranked')
    
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(x_labels)
    
    axs[1].set_xlabel('Average Detection Score in Bin (and count)')
    axs[1].set_ylabel('NDCG@5')
    axs[1].set_title('NDCG@5 by Detection Score Bin')
    axs[1].set_ylim(0, 1.0)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_score_bins.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Performance improvement by detection score
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvement percentages
    bin_metrics['top1_improvement'] = (bin_metrics['reranked_top1'] - bin_metrics['original_top1']) / bin_metrics['original_top1'] * 100
    bin_metrics.loc[bin_metrics['original_top1'] == 0, 'top1_improvement'] = 0  # Handle division by zero
    
    bin_metrics['top3_improvement'] = (bin_metrics['reranked_top3'] - bin_metrics['original_top3']) / bin_metrics['original_top3'] * 100
    bin_metrics.loc[bin_metrics['original_top3'] == 0, 'top3_improvement'] = 0  # Handle division by zero
    
    bin_metrics['ndcg_improvement'] = (bin_metrics['reranked_ndcg'] - bin_metrics['original_ndcg']) / bin_metrics['original_ndcg'] * 100
    bin_metrics.loc[bin_metrics['original_ndcg'] == 0, 'ndcg_improvement'] = 0  # Handle division by zero
    
    # Plot improvements
    ax.plot(bin_metrics['detection_score'], bin_metrics['top1_improvement'], 'o-', label='Top-1 Improvement')
    ax.plot(bin_metrics['detection_score'], bin_metrics['top3_improvement'], 's-', label='Top-3 Improvement')
    ax.plot(bin_metrics['detection_score'], bin_metrics['ndcg_improvement'], '^-', label='NDCG@5 Improvement')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Average Detection Score')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Ranking Improvement by Detection Score')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_score_improvement.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Save binned data to CSV
    bin_metrics.to_csv(os.path.join(output_dir, 'detection_score_bins.csv'), index=False)

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
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, bin_metrics, output_dir)
    
    print("\nAnalysis complete. Output files have been saved to the output directory.")

if __name__ == "__main__":
    main() 