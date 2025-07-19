import pandas as pd
import os
import matplotlib.pyplot as plt

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
colors = ["blue", "orange", "pink", "green"]

# Load CSV files Reconstruction loss 
plots_rec = {
    0: ("/runs/vanilla-vae/15_epochs/plots/reconstruction_loss_mean.csv", "/runs/vanilla-vae/15_epochs/plots/reconstruction_loss_stdev.csv", "Vanilla"),
    1: ("/runs/beta-vae/beta_0.3_incr/plots/reconstruction_loss_mean.csv", "/runs/beta-vae/beta_0.3_incr/plots/reconstruction_loss_stdev.csv", "β=0.3 incremental"),
    2: ("/runs/beta-vae/beta_5/plots/reconstruction_loss_mean.csv", "/runs/beta-vae/beta_5/plots/reconstruction_loss_stdev.csv", "β=5"),
    3: ("/runs/beta-vae/beta_20/plots/reconstruction_loss_mean.csv", "/runs/beta-vae/beta_20/plots/reconstruction_loss_stdev.csv", "β=20"),
}

plots_kl = {
    0: ("/runs/vanilla-vae/15_epochs/plots/kl_div_mean.csv", "/runs/vanilla-vae/15_epochs/plots/kl_div_stdev.csv", "Vanilla"),
    1: ("/runs/beta-vae/beta_0.3_incr/plots/kl_div_mean.csv", "/runs/beta-vae/beta_0.3_incr/plots/kl_div_stdev.csv", "β=0.3 incremental"),
    2: ("/runs/beta-vae/beta_5/plots/kl_div_mean.csv", "/runs/beta-vae/beta_5/plots/kl_div_stdev.csv", "β=5"),
    3: ("/runs/beta-vae/beta_20/plots/kl_div_mean.csv", "/runs/beta-vae/beta_20/plots/kl_div_stdev.csv", "β=20"),
}



for plot in zip(colors, plots_kl.values()):

    mean = pd.read_csv(root_dir + plot[1][0])
    stev = pd.read_csv(root_dir +  plot[1][1])

    # Get the values from the CSV files IPPO
    steps = mean["Step"]
    mean_values = mean["Value"]
    stev_values = stev["Value"]
    stev_max = [stev_values[i]+mean_values[i] for i in range(len(mean_values))]
    stev_min = [mean_values[i]-stev_values[i] for i in range(len(mean_values))]

    # Plot the mean line
    plt.plot(steps, mean_values, label=plot[1][2], color=plot[0])

    # Plot the shaded region
    plt.fill_between(steps, stev_min, stev_max, color=plot[0], alpha=0.2)

# Customize the plot
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Reconstruction Loss (per-batch)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
