import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib
#matplotlib.use('Agg')


base_dir = '/MPC_DynamicSys/code/cart_pole_diffusion_based_on_MPD/training_data/CartPole-NMPC'
J_DATA_FILENAME = 'j_ini_10x20_noise_20_step_80_hor_64.pt'

j_load = torch.load(os.path.join(base_dir, J_DATA_FILENAME))
j_load_np = j_load.numpy()
print('data load')

j_load_np = np.log(j_load_np)
j_load_np = 2*(j_load_np - j_load_np.min())/(j_load_np.max()-j_load_np.min())-1
# j_load_np = (j_load_np - np.mean(j_load_np))/np.std(j_load_np)


# calculate feature
mean_value = np.mean(j_load_np)
median_value = np.median(j_load_np)
q25 = np.percentile(j_load_np, 25)  # 25th percentile (1st quartile)
q75 = np.percentile(j_load_np, 75) 
std_value_np = np.std(j_load_np)

print("mean_value=",mean_value)
print("median_value=",median_value)
print("q25=",q25)
print("q75=",q75)
print("std_value_np=",std_value_np)

# Plot the distribution using a histogram (Matplotlib)
plt.figure(figsize=(10, 6))
sns.histplot(j_load_np, bins=100, kde=True, color='blue', alpha=0.6)

# Add vertical lines for the statistics
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='--', label=f'Median: {median_value:.2f}')
plt.axvline(q25, color='orange', linestyle='--', label=f'25th Percentile (Q1): {q25:.2f}')
plt.axvline(q75, color='purple', linestyle='--', label=f'75th Percentile (Q3): {q75:.2f}')

# Add labels and title
plt.title('Data Distribution with Mean, Median, 25th, and 75th Percentiles')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()

figure_name = 'cost_distribution_log_minmax.png'
plt.savefig(figure_name)
print("save")