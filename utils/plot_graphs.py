import pandas as pd
import os
import matplotlib.pyplot as plt

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
color1 = "blue"

# Load CSV files Reconstruction loss 
mean = pd.read_csv(root_dir + "/runs/vanilla-vae/15_epochs/plots/reconstruction_loss_mean.csv")
stev = pd.read_csv(root_dir + "/runs/vanilla-vae/15_epochs/plots/reconstruction_loss_stdev.csv")

# Get the values from the CSV files IPPO
steps = mean["Step"]
mean_values = mean["Value"]
stev_values = stev["Value"]
stev_max = [stev_values[i]+mean_values[i] for i in range(len(mean_values))]
stev_min = [mean_values[i]-stev_values[i] for i in range(len(mean_values))]


# Plot the mean line
plt.plot(steps, mean_values, label="Reconstruction Loss (per-batch)", color=color1)

# Plot the shaded region
plt.fill_between(steps, stev_min, stev_max, color=color1, alpha=0.2)

# Customize the plot
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Reconstruction Loss (vanilla VAE)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
