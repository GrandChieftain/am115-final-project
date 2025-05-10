#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
from analyze_rankings import analyze_rankings

def create_bar_chart(metrics, total_processed, output_file=None):
    """Create a bar chart comparing original and reranked metrics"""
    # Set up the metrics and labels
    metrics_names = ['Top-1', 'Top-2', 'Top-3', 'NDCG@5']
    original_values = [metrics['original']['top1'], metrics['original']['top2'], 
                       metrics['original']['top3'], metrics['original']['ndcg5']]
    reranked_values = [metrics['reranked']['top1'], metrics['reranked']['top2'],
                       metrics['reranked']['top3'], metrics['reranked']['ndcg5']]
    
    # Calculate improvement percentages
    improvements = []
    for o, r in zip(original_values, reranked_values):
        if o > 0:
            improvements.append(((r - o) / o) * 100)
        else:
            improvements.append(0)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Bar chart width
    width = 0.35
    
    # X positions for bars
    x = np.arange(len(metrics_names))
    
    # Create the bar charts on the first subplot
    rects1 = ax1.bar(x - width/2, original_values, width, label='Original Google Lens')
    rects2 = ax1.bar(x + width/2, reranked_values, width, label='Reranked')
    
    # Add some text for labels, title and axes ticks
    ax1.set_ylabel('Accuracy/Score')
    ax1.set_title(f'Ranking Performance Metrics (n={total_processed})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    
    # Add value labels on the bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    
    # Create the improvement percentage chart
    colors = ['red' if x < 0 else 'green' for x in improvements]
    ax2.bar(metrics_names, improvements, color=colors)
    ax2.set_ylabel('Percent Change (%)')
    ax2.set_title('Relative Performance Change')
    
    # Add horizontal line at 0%
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(improvements):
        ax2.text(i, v + np.sign(v) * 1, f"{v:.1f}%", ha='center', fontsize=9)
    
    # Adjust layout and save if requested
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_file}")
    
    plt.show()

def main():
    matches_file = "../results/matches.csv"
    results_dir = "../results"
    
    print(f"Analyzing rankings for {matches_file}...")
    metrics, total_processed = analyze_rankings(matches_file, results_dir)
    
    # Create the visualization
    create_bar_chart(metrics, total_processed, output_file="ranking_performance.png")
    
    # Print text summary as well
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
    
    print("\nImprovement:")
    for metric in ['top1', 'top2', 'top3', 'ndcg5']:
        orig = metrics['original'][metric]
        rerank = metrics['reranked'][metric]
        rel_improvement = ((rerank - orig) / orig) * 100 if orig > 0 else float('inf')
        abs_improvement = rerank - orig
        print(f"  {metric}: {rel_improvement:.2f}% (absolute: {abs_improvement:.4f})")

if __name__ == "__main__":
    main() 