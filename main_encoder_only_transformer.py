import numpy as np
import pynncml as pnc
import torch
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy
from sklearn import metrics
from torch.utils.data import Subset
import os
import torch.nn as nn
from io import StringIO
import sys

from encoder_only_transformer.model import EncoderOnlyTransformer

xy_min = [1.29e6, 0.565e6]  # Link Region
xy_max = [1.34e6, 0.5875e6]
time_slice = slice("2015-06-01", "2015-06-10")  # Time Interval

samples_type = "min_max"  # Options: "instantaneous", "min_max"
sampling_interval_in_sec = 180 # Options: 10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900

if samples_type == "min_max":
    rnn_input_size = 4  # MRSL, mRSL, MTSL, mTSL
    sampling_interval_in_sec = 900
elif samples_type == "instantaneous":
    rnn_input_size = 2 * (900 // sampling_interval_in_sec)


# Set output directory based on sampling configuration (lab computer path)
base_output_dir = "/Users/barakmachlev/Documents/Thesis/Transformer"
if samples_type == "instantaneous":
    output_dir = os.path.join(base_output_dir, f"Instantaneous_{sampling_interval_in_sec}_sec")
else:
    output_dir = os.path.join(base_output_dir, "Max_Min")

os.makedirs(output_dir, exist_ok=True)

dataset = pnc.datasets.loader_open_mrg_dataset(xy_min = xy_min,
                                               xy_max = xy_max,
                                               time_slice = time_slice,
                                               samples_type = samples_type,
                                               sampling_interval_in_sec = sampling_interval_in_sec)

plt.figure(1)
dataset.link_set.plot_links(scale=True, scale_factor=1.0)
plt.grid()
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.title("CML Link Map")
figure_name = "CML_Link_Map.png"
save_path = os.path.join(output_dir, figure_name)
plt.savefig(save_path)
plt.tight_layout()
print(f"✅ Figure saved to {save_path}")
plt.show(block=False)
plt.close()

plt.figure(2)
rg = np.stack([p.data_array for p in dataset.point_set]).flatten()
param = scipy.stats.expon.fit(rg)
exp_gamma = param[1]
print("Rain Rate Statistics")
print("Mean[mm/hr]:", np.mean(rg))
print("Std[mm/hr]:", np.std(rg))
print("Parentage of wet samples", 100 * np.sum(rg > 0) / rg.size)
print("Parentage of dry samples", 100 * np.sum(rg == 0) / rg.size)
print("Exponential Distribution Parameters:", param)
_ = plt.hist(rg, bins=100, density=True)
plt.plot(np.linspace(0, np.max(rg), 100), scipy.stats.expon.pdf(np.linspace(0, np.max(rg), 100), *param))
plt.grid()
plt.xlabel("Rain Rate[mm/hr]")
plt.ylabel("Density")
plt.title("Rain Rate Histogram")
figure_name = "Rain_Rate_Histogram.png"
save_path = os.path.join(output_dir, figure_name)
plt.savefig(save_path)
print(f"✅ Figure saved to {save_path}")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

batch_size = 16
rnn_n_features = 128  # @param{type:"integer"}
n_layers = 2  # @param{type:"integer"}
lr = 1e-4  # @param{type:"number"}
weight_decay = 1e-4  # @param{type:"number"}
rnn_type = pnc.neural_networks.RNNType.GRU  # RNN Type
n_epochs = 12  # @param{type:"integer"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)
if device.type == "cuda":
    print("  - GPU Name:", torch.cuda.get_device_name(0))
else:
    print("  - Running on CPU")


# Configurable target link
target_link = 0  # 👈 Change this to choose which link must appear last in validation

# Fixed split for 132 links: 106 for training, 26 for validation (link 0 last in val)
assert len(dataset) == 132, "Unexpected number of links — expected 132!"
all_indices = np.arange(132)
np.random.seed(42)  # Ensure repeatable split
np.random.shuffle(all_indices)

# Remove 0 (we will force it into last val index)
all_indices = all_indices[all_indices != target_link]

train_indices = all_indices[:106].tolist()  # 106
val_indices = all_indices[106:].tolist()  # 26
val_indices.append(target_link)  # Ensure link 0 is last in validation

training_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, val_indices)

data_loader = torch.utils.data.DataLoader(training_dataset, batch_size)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size)

print(f"✅ Link {target_link} is ensured as last in validation set.")

# 🔍 Sanity checks
train_ids = set(training_dataset.indices)
val_ids = set(validation_dataset.indices)

assert train_ids.isdisjoint(val_ids), "❌ Overlap between training and validation sets!"
assert train_ids.union(val_ids) == set(range(132)), "❌ Some link indices missing!"
assert validation_dataset.indices[-1] == target_link, "❌ Link 0 is not last in validation set!"


window_size = 32
metadata_n_features = 32
dynamic_dim = 4
metadata_dim = 2
d_model = 512
dropout = 0.1
um_encoder_layers=4
h=8
model = EncoderOnlyTransformer(dynamic_input_size=dynamic_dim,
                                   metadata_input_size=metadata_dim,
                                   d_model=d_model,
                                   metadata_n_features=metadata_n_features,
                                   window_size=window_size,
                                   dropout=dropout,
                                   num_encoder_layers=um_encoder_layers,
                                   h=h).to(device)

from pynncml.metrics.results_accumlator import ResultsAccumulator, AverageMetric, GroupAnalysis

opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
ra = ResultsAccumulator()
am = AverageMetric()


loss_est = 0
epoch_losses=[]

# Train model if weights do not exist
model.train()
for epoch in tqdm(range(n_epochs)): # Repeat the whole training process again
    am.clear()
    for rain_rate, rsl, tsl, metadata in data_loader: # for loop for true batches - each batch has batch_size links
        m_step = math.floor(rain_rate.shape[1] / window_size)
        for step in range(m_step): # for loop for sliding windows in time (a.k.a mini-batches / chunks)
            opt.zero_grad()  # Zero gradients
            # Perform sliding window in the CML time series.
            _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
            _rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            _tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
            # Forward pass of the model
            Transformer_encoder_output = model(torch.cat([_rsl, _tsl], dim=-1), metadata.to(device))
            rain_hat = Transformer_encoder_output[:, :, 0]
            # Compute weighted RMSE
            wet_mask = (_rr > 0).float()
            dry_mask = (_rr == 0).float()

            wet_weight = 10.0
            dry_weight = 1.0

            weights = wet_weight * wet_mask + dry_weight * dry_mask

            mse = (weights * (rain_hat - _rr) ** 2).mean()
            loss = torch.sqrt(mse)            # Take the derivative w.r.t. model parameters $\Theta$

            loss.backward()
            opt.step()
            am.add_results(loss=loss.item())  # Log results to average.

    avg_loss = am.get_results("loss")
    ra.add_results(loss=avg_loss)
    epoch_losses.append(avg_loss)

    print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.6f}")



plt.plot(epoch_losses, label="Training RMSE Loss")
plt.grid()
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss per Epoch ({samples_type}.png)")
figure_name = f"loss_plot_over_epochs_{samples_type}.png"
save_path = os.path.join(output_dir, figure_name)
plt.savefig(save_path)
print(f"✅ Figure saved to {save_path}")
plt.show(block=False)
plt.pause(10)
plt.close()