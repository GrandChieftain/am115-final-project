import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the category metrics data
df = pd.read_csv('../results/category_metrics.csv')

# Sort categories by original NDCG@5 in descending order
df = df.sort_values(by='Original NDCG@5', ascending=False)

# Get categories and metrics
categories = df['Category']
original_ndcg = df['Original NDCG@5']
reranked_ndcg = df['Reranked NDCG@5']
kendall_tau = df['Kendall Tau']

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

# Set width of bars
bar_width = 0.35
index = np.arange(len(categories))

# Create bars
original_bars = ax1.bar(index - bar_width/2, original_ndcg, bar_width, 
                        color='blue', alpha=0.8, label='Original Google Lens')
reranked_bars = ax1.bar(index + bar_width/2, reranked_ndcg, bar_width, 
                        color='red', alpha=0.8, label='Reranked')

# Add values on top of bars
for i, v in enumerate(original_ndcg):
    ax1.text(i - bar_width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
for i, v in enumerate(reranked_ndcg):
    ax1.text(i + bar_width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

# Customize plot
ax1.set_xlabel('Clothing Category', fontsize=14)
ax1.set_ylabel('NDCG@5 Score', fontsize=14)
ax1.set_title('Ranking Accuracy (NDCG@5) by Clothing Category', fontsize=16, fontweight='bold')
ax1.set_xticks(index)
ax1.set_xticklabels(categories, rotation=45, ha='right')
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=12, loc='lower right')

# Add count annotations
for i, cat in enumerate(categories):
    count = df[df['Category'] == cat]['Count'].values[0]
    ax1.text(i, 0.05, f'n={count}', ha='center', va='bottom', fontsize=9)

# Plot Kendall Tau in the second subplot
bars = ax2.bar(index, kendall_tau, color=[
    'green' if x < 0 else 'orange' for x in kendall_tau
], alpha=0.8)

# Add values on top of bars
for i, v in enumerate(kendall_tau):
    position = v + 0.02 if v >= 0 else v - 0.08
    ax2.text(i, position, f'{v:.2f}', ha='center', va='center', fontsize=9, 
             color='black' if v >= 0 else 'white')

# Customize Kendall Tau plot
ax2.set_xlabel('Clothing Category', fontsize=14)
ax2.set_ylabel('Kendall Tau', fontsize=14)
ax2.set_title('Kendall Tau by Clothing Category (negative = less reordering)', fontsize=14)
ax2.set_xticks(index)
ax2.set_xticklabels(categories, rotation=45, ha='right')
ax2.axhline(y=0, color='grey', linestyle='--')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add an annotation explaining Kendall Tau
ax2.text(len(categories)-1, 0.2, 
         "Kendall Tau measures ranking changes:\n" +
         "Higher values = more reordering\n" +
         "Negative values = reversed ordering",
         ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('category_comparison.png', dpi=300)
print("Plot saved as 'category_comparison.png'") 