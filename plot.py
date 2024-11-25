import matplotlib.pyplot as plt
import seaborn as sns
from data import *

# Apply a sophisticated Seaborn style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})

# Define color palette for clarity
colors = {
    "split_learning": "#1f77b4",
    "split_fed": "#ff7f0e",
    "sharding_split_fed": "#2ca02c",
    "bl_split_fed": "#d62728",
}

def plot_times():
    plt.plot(range(1, len(split_learning_times) + 1), split_learning_times, label="Split Learning", color=colors["split_learning"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(split_fed_times) + 1), split_fed_times, label="SplitFed Learning", color=colors["split_fed"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(sharding_split_fed_times) + 1), sharding_split_fed_times, label="Sharding SplitFed", color=colors["sharding_split_fed"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(bl_split_fed_times) + 1), bl_split_fed_times, label="Blockchain SplitFed", color=colors["bl_split_fed"], linewidth=2, antialiased=True)

def plot_normal_9():
    plt.plot(range(1, len(split_learning) + 1), split_learning, label="Split Learning", color=colors["split_learning"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(split_fed) + 1), split_fed, label="SplitFed Learning", color=colors["split_fed"], linewidth=2)
    plt.plot(range(1, len(sharding_split_fed) + 1), sharding_split_fed, label="Sharding SplitFed", color=colors["sharding_split_fed"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(bl_split_fed) + 1), bl_split_fed, label="Blockchain SplitFed", color=colors["bl_split_fed"], linewidth=2, antialiased=True)

def plot_attack_9():
    plt.plot(range(1, len(attack_split_learning) + 1), attack_split_learning, label="Attacked Split Learning", color=colors["split_learning"], linestyle="dashed", linewidth=2, antialiased=True)
    plt.plot(range(1, len(attack_split_fed) + 1), attack_split_fed, label="Attacked SplitFed Learning", color=colors["split_fed"], linestyle="dashed", linewidth=2, antialiased=True)
    plt.plot(range(1, len(attack_sharding_split_fed) + 1), attack_sharding_split_fed, label="Attacked Sharding SplitFed", color=colors["sharding_split_fed"], linestyle="dashed", linewidth=2, antialiased=True)
    plt.plot(range(1, len(attack_bl_split_fed) + 1), attack_bl_split_fed, label="Attacked Blockchain SplitFed", color=colors["bl_split_fed"], linestyle="dashed", linewidth=2, antialiased=True)

def plot_all_9():
    plot_normal_9()
    plot_attack_9()

def plot_normal_36():
    plt.plot(range(1, len(split_learning_36) + 1), split_learning_36, label="Split Learning", color=colors["split_learning"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(split_fed_36) + 1), split_fed_36, label="SplitFed Learning", color=colors["split_fed"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(sharding_split_fed_36) + 1), sharding_split_fed_36, label="Sharding SplitFed", color=colors["sharding_split_fed"], linewidth=2, antialiased=True)
    plt.plot(range(1, len(bl_split_fed_36) + 1), bl_split_fed_36, label="Blockchain SplitFed", color=colors["bl_split_fed"], linewidth=2, antialiased=True)

def plot_attack_36():
    plt.plot(range(1, len(attack_split_learning_36) + 1), attack_split_learning_36, label="Attacked Split Learning", color=colors["split_learning"], linestyle="dashed", linewidth=2, antialiased=True)
    plt.plot(range(1, len(attack_split_fed_36) + 1), attack_split_fed_36, label="Attacked SplitFed Learning", color=colors["split_fed"], linestyle="dashed", linewidth=2, antialiased=True)
    plt.plot(range(1, len(attack_sharding_split_fed_36) + 1), attack_sharding_split_fed_36, label="Attacked Sharding SplitFed", color=colors["sharding_split_fed"], linestyle="dashed", linewidth=2, antialiased=True)
    plt.plot(range(1, len(attack_bl_split_fed_36) + 1), attack_bl_split_fed_36, label="Attacked Blockchain SplitFed", color=colors["bl_split_fed"], linestyle="dashed", linewidth=2, antialiased=True)

def plot_all_36():
    plot_normal_36()
    plot_attack_36()


def plot_results(plot_function, num_nodes, image_name, x_max):
    plt.figure(figsize=(12, 6))
    plot_function()
    
    # Formatting
    plt.title(f'Performance Comparison of Learning Approaches ({num_nodes} Nodes)', fontsize=18, fontweight='bold')
    plt.xlabel('Communication Rounds', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    
    # Ticks
    plt.xticks([1] + list(range(5, x_max + 5, 5)), fontsize=16)
    plt.yticks(fontsize=16)
    
    # Grid and Legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.6)
    plt.legend(title="Methods", title_fontsize=16, loc='best', fontsize=14)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"./figures/{image_name}.png", dpi=300)
    plt.close()  # Close figure to manage memory

# Plot Transmission Costs for Learning Approaches
def plot_transmission(num_nodes, image_name, x_max):
    plt.figure(figsize=(12, 6))
    plot_times()
    
    # Titles and Labels
    plt.title(f'Transmission Costs of Learning Approaches ({num_nodes} Nodes)', fontsize=18, fontweight='bold')
    plt.xlabel('Communication Rounds', fontsize=16)
    plt.ylabel('Cumulative Time', fontsize=16)
    
    # Ticks
    plt.xticks([1] + list(range(5, x_max + 5, 5)), fontsize=16)
    plt.yticks(fontsize=14)
    
    # Grid and Legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.6)
    plt.legend(title="Methods", title_fontsize=16, loc='upper left', fontsize=14)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"./figures/{image_name}.png", dpi=500)
    plt.close()


plot_results(plot_normal_9, num_nodes=9, image_name="normal_9", x_max=60)
plot_results(plot_attack_9, num_nodes=9, image_name="attack_9", x_max=60)
plot_results(plot_all_9, num_nodes=9, image_name="all_9", x_max=60)
plot_results(plot_normal_36, num_nodes=36, image_name="normal_36", x_max=30)
plot_results(plot_attack_36, num_nodes=36, image_name="attack_36", x_max=30)
plot_results(plot_all_36, num_nodes=36, image_name="all_36", x_max=30)
plot_transmission(num_nodes=36, image_name="times", x_max=30)