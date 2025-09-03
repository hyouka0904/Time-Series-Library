import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV file
df = pd.read_csv('experiment_results.csv')

# Set figure parameters
fig, ax = plt.subplots(figsize=(10, 6))

# Set model names and count
models = df['model'].tolist()
n_models = len(models)

# Set bar width and positions
bar_width = 0.35
x = np.arange(2)  # Two groups: MSE and MAE

# Assign different colors for each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot MSE and MAE for each model
for i, (model, color) in enumerate(zip(models, colors)):
    model_data = df[df['model'] == model]
    values = [model_data['mse'].values[0], model_data['mae'].values[0]]

    # Calculate positions for each bar
    positions = x + (i - 1) * bar_width / n_models

    # Draw bars
    bars = ax.bar(positions, values, bar_width / n_models,
                  label=model, color=color, alpha=0.8)

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

# Set x-axis tick labels
ax.set_xticks(x)
ax.set_xticklabels(['MSE', 'MAE'])

# Set title and labels
ax.set_xlabel('Evaluation Metrics', fontsize=12)
ax.set_ylabel('Values', fontsize=12)
ax.set_title('Model Performance Comparison: MSE vs MAE', fontsize=14, fontweight='bold')

# Add legend
ax.legend(title='Model', loc='upper right')

# Add grid lines
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Print performance summary
print("Model Performance Summary:")
print("-" * 50)
for _, row in df.iterrows():
    print(f"{row['model']}:" )
    print(f"  MSE: {row['mse']:.6f}")
    print(f"  MAE: {row['mae']:.6f}")
    print("-" * 50)
