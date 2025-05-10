import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Read the detection score metrics data
df = pd.read_csv('../results/detection_score_metrics.csv')

# Define bins for detection scores
bins = np.linspace(0.4, 1.0, 7)  # 6 bins from 0.4 to 1.0
df['detection_score_bin'] = pd.cut(df['detection_score'], bins)

# Calculate mean metrics for each bin
bin_metrics = df.groupby('detection_score_bin').agg({
    'original_top1': 'mean',
    'original_top3': 'mean',
    'original_ndcg': 'mean',
    'reranked_top1': 'mean',
    'reranked_top3': 'mean',
    'reranked_ndcg': 'mean'
}).reset_index()

# Get bin midpoints for x-axis
bin_metrics['bin_midpoint'] = bin_metrics['detection_score_bin'].apply(lambda x: x.mid)

# Configure plot
plt.figure(figsize=(12, 8))

# Plot original metrics
plt.plot(bin_metrics['bin_midpoint'], bin_metrics['original_top1'], 'o-', color='blue', linewidth=2, label='Original Top-1')
plt.plot(bin_metrics['bin_midpoint'], bin_metrics['original_top3'], 's-', color='darkblue', linewidth=2, label='Original Top-3')
plt.plot(bin_metrics['bin_midpoint'], bin_metrics['original_ndcg'], '^-', color='royalblue', linewidth=2, label='Original NDCG@5')

# Plot reranked metrics
plt.plot(bin_metrics['bin_midpoint'], bin_metrics['reranked_top1'], 'o--', color='red', linewidth=2, label='Reranked Top-1')
plt.plot(bin_metrics['bin_midpoint'], bin_metrics['reranked_top3'], 's--', color='darkred', linewidth=2, label='Reranked Top-3')
plt.plot(bin_metrics['bin_midpoint'], bin_metrics['reranked_ndcg'], '^--', color='firebrick', linewidth=2, label='Reranked NDCG@5')

# Format the plot
plt.xlabel('Detection Score', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Detection Score vs. Ranking Accuracy', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Format y-axis as percentage
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

# Set nice limits
plt.ylim(0, 1.05)
plt.xlim(0.39, 1.01)

# Add annotations
plt.text(0.42, 0.05, f"Total images analyzed: {len(df)}", fontsize=10)

# Save the plot
plt.tight_layout()
plt.savefig('detection_score_vs_accuracy.png', dpi=300)
print("Plot saved as 'detection_score_vs_accuracy.png'") 