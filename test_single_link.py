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
from pynncml.neural_networks import InputNormalizationConfig

time_slice = slice("2015-06-01", "2015-08-31")  # Time Interval

samples_type = "min_max"  # Options: "instantaneous", "min_max"
sampling_interval_in_sec = 450 # Options: 10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900

if samples_type == "min_max":
    rnn_input_size = 4  # MRSL, mRSL, MTSL, mTSL
    sampling_interval_in_sec = 900
elif samples_type == "instantaneous":
    rnn_input_size = 2 * (900 // sampling_interval_in_sec)

# Set output directory based on sampling configuration
output_dir = "/Users/barakmachlev/Documents/Thesis/Influence_of_sampling_intervals_Results/Single_link/Max_Min"
if samples_type == "instantaneous":
    output_dir = f"/Users/barakmachlev/Documents/Thesis/Influence_of_sampling_intervals_Results/Single_link/Instantaneous_{sampling_interval_in_sec}_sec"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

dataset = pnc.datasets.loader_open_mrg_dataset(time_slice = time_slice,
                                               samples_type = samples_type,
                                               sampling_interval_in_sec = sampling_interval_in_sec)


plt.figure(1)
dataset.link_set.plot_links(scale=True, scale_factor=1.0)
plt.grid()
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.title("CML Link Map")
plt.show(block=False)
plt.pause(2)
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
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

window_size = 32  # @param{type:"integer"}
batch_size = window_size  # @param{type:"integer"}
rnn_n_features = 128  # @param{type:"integer"}
metadata_n_features = 32  # @param{type:"integer"}
n_layers = 2  # @param{type:"integer"}
lr = 1e-4  # @param{type:"number"}
weight_decay = 1e-4  # @param{type:"number"}
rnn_type = pnc.neural_networks.RNNType.GRU  # RNN Type
n_epochs = 100  # @param{type:"integer"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
data_loader = torch.utils.data.DataLoader(training_dataset, batch_size)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size)

'''
# Extract the first link from the dataset
link = dataset.link_set.link_list[0]

# Get full time-aligned data for that link
rain, rsl, tsl, metadata = link.data_alignment()  # Shapes: [T], [T,2], [T,2], [2]

# Compute split index (80% train / 20% val)
n_samples = rain.shape[0]
split_index = int(0.8 * n_samples)

# Slice time series
rain_train, rain_val = rain[:split_index], rain[split_index:]
rsl_train, rsl_val = rsl[:split_index], rsl[split_index:]
tsl_train, tsl_val = tsl[:split_index], tsl[split_index:]

# Wrap each slice into a dataset returning (rain, rsl, tsl, metadata)
class SingleLinkTimeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, rain, rsl, tsl, metadata):
        self.rain = rain
        self.rsl = rsl
        self.tsl = tsl
        self.metadata = metadata

    def __len__(self):
        return self.rain.shape[0]  # T samples

    def __getitem__(self, idx):
        return self.rain[idx], self.rsl[idx], self.tsl[idx], self.metadata

# Create train/val datasets
training_dataset = SingleLinkTimeWindowDataset(rain_train, rsl_train, tsl_train, metadata)
validation_dataset = SingleLinkTimeWindowDataset(rain_val, rsl_val, tsl_val, metadata)

data_loader = torch.utils.data.DataLoader(training_dataset, batch_size)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size)



# 🔍 DEBUG: Print shape of input tensors before normalization
print(f"🔢 rsl_train.shape: {rsl_train.shape}")
print(f"🔢 tsl_train.shape: {tsl_train.shape}")
print(f"🔢 metadata.shape: {metadata.shape}")

# ✅ Compute dynamic mean/std from RSL + TSL
_data = torch.cat([torch.from_numpy(rsl_train), torch.from_numpy(tsl_train)], dim=-1)
_data = _data.reshape(-1, rnn_input_size)
mean_dynamic = _data.mean(dim=0).cpu().numpy().reshape(1, 1, -1)
std_dynamic = _data.std(dim=0).cpu().numpy().reshape(1, 1, -1)
std_dynamic[std_dynamic < 1e-6] = 1.0  # Prevent division by zero

# ✅ Compute metadata mean/std
mean_meta = torch.from_numpy(metadata).reshape(1, -1).cpu().numpy()
std_meta = np.ones_like(mean_meta)  # Avoid zero std for constant metadata

# 🧾 Print computed stats
print("📊 Dynamic mean:", mean_dynamic)
print("📊 Dynamic std:", std_dynamic)
print("📊 Metadata mean:", mean_meta)
print("📊 Metadata std:", std_meta)

# ✅ Wrap into InputNormalizationConfig
normalization_cfg = InputNormalizationConfig(mean_dynamic, std_dynamic, mean_meta, std_meta)


model = pnc.scm.rain_estimation.two_step_network(n_layers=n_layers,  # Number of RNN layers
                                                 rnn_type=rnn_type,  # Type of RNN (GRU, LSTM)
                                                 normalization_cfg=normalization_cfg,
                                                 rnn_input_size = rnn_input_size,  # 3 + 3 (RSL + TSL)
                                                 rnn_n_features=rnn_n_features,  # Number of features in the RNN
                                                 metadata_input_size=2,  # Number of metadata features
                                                 metadata_n_features=metadata_n_features, # Number of features in the metadata
                                                 pretrained=False).to(device)  # Pretrained model is set to False to train the model from scratch.

class RegressionLoss(torch.nn.Module):
    def __init__(self, in_gamma, gamma_s=0.9):
        super(RegressionLoss, self).__init__()
        self.in_gamma = in_gamma
        self.gamma_s = gamma_s

    def forward(self, input, target):
        delta = (target - input) ** 2
        w = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)
        return torch.sum(torch.mean(w * delta, dim=0))

from pynncml.metrics.results_accumlator import ResultsAccumulator, AverageMetric, GroupAnalysis

opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
ra = ResultsAccumulator()
am = AverageMetric()

model_path = os.path.join(output_dir, "trained_model.pth")

if os.path.exists(model_path):
    # Load model if weights already exist
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from: {model_path} — skipping training and loss plotting")

else:

    print("🟡 No saved weights found — starting training")

    loss_function_rain_est = RegressionLoss(exp_gamma)
    loss_function_wet_dry = torch.nn.BCELoss()

    # Find lambda values such that at initialization both loss will be equal:
    loss_est = 0
    loss_detection = 0
    with torch.no_grad():
        for rain_rate, rsl, tsl, metadata in data_loader:
            state = model.init_state(batch_size=1)

            # Adjust dimensions for single link
            rsl = rsl.unsqueeze(0)  # [1, T, F]
            tsl = tsl.unsqueeze(0)  # [1, T, F]
            rain_rate = rain_rate.unsqueeze(0)  # [1, T]
            metadata = metadata[0].unsqueeze(0).to(device)  # Result: [1, 2]

            #print("📥 Input sizes:")
            #print("rsl:", rsl.size(), "min:", rsl.min().item(), "max:", rsl.max().item())
            #print("tsl:", tsl.size(), "min:", tsl.min().item(), "max:", tsl.max().item())
            #print("rain_rate:", rain_rate.size(), "min:", rain_rate.min().item(), "max:", rain_rate.max().item())
            #print("metadata:", metadata.size(), "values:", metadata)

            rain_estimation_detection, state = model(torch.cat([rsl, tsl], dim=-1), metadata.to(device), state.detach())

            rain_hat = rain_estimation_detection[:, :, 0]
            rain_detection = rain_estimation_detection[:, :, 1]

            loss_est += loss_function_rain_est(rain_hat, rain_rate)
            loss_detection += loss_function_wet_dry(rain_detection, (rain_rate > 0.1).float())
    lambda_value = loss_detection / loss_est

    # Train model if weights do not exist
    model.train()
    for epoch in tqdm(range(n_epochs)):  # Repeat the whole training process again
        am.clear()
        for rain_rate, rsl, tsl, metadata in data_loader:
            state = model.init_state(batch_size = 1)
            opt.zero_grad()

            # Adjust dimensions for single link
            rsl = rsl.unsqueeze(0)  # [1, T, F]
            tsl = tsl.unsqueeze(0)  # [1, T, F]
            rain_rate = rain_rate.unsqueeze(0)  # [1, T]
            metadata = metadata[0].unsqueeze(0).to(device)  # Result: [1, 2]

            rain_estimation_detection, state = model(torch.cat([rsl, tsl], dim=-1), metadata.to(device), state.detach())
            rain_hat = rain_estimation_detection[:, :, 0]
            rain_detection = rain_estimation_detection[:, :, 1]

            loss_est = loss_function_rain_est(rain_hat, rain_rate)
            loss_detection = loss_function_wet_dry(rain_detection, (rain_rate > 0.1).float())
            loss = lambda_value * loss_est + loss_detection

            loss.backward()
            opt.step()

            am.add_results(loss=loss.item(), loss_est=loss_est.item(), loss_detection=loss_detection.item())

        ra.add_results(loss=am.get_results("loss"), loss_est=am.get_results("loss_est"),
                       loss_detection=am.get_results("loss_detection"))

    # Save trained weights
    torch.save(model.state_dict(), model_path)
    print(f"✅ Weights saved to: {model_path}")

    plt.plot(ra.get_results("loss"), label="Total Loss")
    plt.plot(ra.get_results("loss_est"), label="Rain Rate Loss")
    plt.plot(ra.get_results("loss_detection"), label="Wet/Dry Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss per Epoch ({samples_type}.png)")
    figure_name = f"loss_plot_over_epochs_{samples_type}.png"
    plt.savefig(os.path.join(output_dir, figure_name))
    plt.show(block=False)
    plt.pause(5)
    plt.close()


model.eval()
ga = GroupAnalysis()
am.clear()

rain_hat_list = []
rain_ref_list = []
detection_list = []

with torch.no_grad():
    state = model.init_state(batch_size=1)
    counter=0
    for rain_rate, rsl, tsl, metadata in val_loader:
        counter=counter+1;
        print(f"\n🔍 Batch {counter}")
        print(f"  rain_rate.shape: {rain_rate.shape}")
        print(f"  rsl.shape: {rsl.shape}")
        print(f"  tsl.shape: {tsl.shape}")
        # Add batch dimension
        rain_rate = rain_rate.unsqueeze(0).to(device)  # [1, T]
        rsl = rsl.unsqueeze(0).to(device)              # [1, T, F]
        tsl = tsl.unsqueeze(0).to(device)              # [1, T, F]
        metadata = metadata[0].unsqueeze(0).to(device) # [1, 2]

        rain_estimation_detection, state = model(torch.cat([rsl, tsl], dim=-1), metadata.to(device), state.detach())
        rain_detection = rain_estimation_detection[:, :, 1]
        rain_hat = rain_estimation_detection[:, :, 0] * torch.round(rain_detection)  # Rain Rate only for wet samples

        print(f"  rain_hat.shape: {rain_hat.shape}")
        print(f"  rain_detection.shape: {rain_detection.shape}")

        rain_hat_list.append(rain_hat.cpu().numpy())
        rain_ref_list.append(rain_rate.cpu().numpy())

        ga.append(rain_ref_list[-1], rain_hat_list[-1])
        detection_list.append(torch.round(rain_detection).cpu().numpy())

        delta = rain_hat.squeeze() - rain_rate.squeeze()
        bias = torch.mean(delta)
        mse = torch.mean(delta ** 2)
        am.add_results(bias=bias.item(), mse=mse.item())
# Flatten arrays
actual = np.concatenate([arr.flatten() for arr in detection_list])
predicted = (np.concatenate([arr.flatten() for arr in rain_ref_list]) > 0.1).astype(float)

confusion_matrix = metrics.confusion_matrix(actual, predicted)
rain_ref_flat = np.concatenate([arr.flatten() for arr in rain_ref_list])
max_rain = np.max(rain_ref_flat)
g_array = np.linspace(0, max_rain, 6)

print("Results Detection:")
print("Validation Results of Two-Step RNN")
print("Accuracy[%]:", 100 * (np.sum(actual == predicted) / actual.size))
print("F1 Score:", metrics.f1_score(actual, predicted))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])

cm_display.plot()
plt.title(f"Confusion Matrix ({samples_type} Sampling)")
figure_name = f"confusion_matrix_{samples_type}.png"
plt.savefig(os.path.join(output_dir, figure_name))
plt.show(block=False)
plt.pause(5)
plt.close()

print("Results Estimation:")
ga.append(rain_rate.squeeze().cpu().numpy(), rain_hat.squeeze().cpu().numpy())

detection_array = np.concatenate([arr.flatten() for arr in detection_list])
rain_ref_array = np.concatenate([arr.flatten() for arr in rain_ref_list])
rain_hat_array = np.concatenate([arr.flatten() for arr in rain_hat_list])
detection_array = np.round(detection_array)

start_idx = 1000
end_idx = 1375

rain_array = rain_ref_array[start_idx:end_idx]
detection_array = detection_array[start_idx:end_idx]

# Plot detection classification
fig, ax = plt.subplots()
x = np.arange(rain_array.shape[0])
ax.plot(x, rain_array, label="Rain")
rain_max = np.max(rain_array)

ax.fill_between(x, rain_max, where=np.logical_or(np.logical_and(detection_array == 1, rain_array > 0),
                                                 np.logical_and(detection_array == 0, rain_array == 0)),
                facecolor='green', alpha=.5, label="Detection")

ax.fill_between(x, rain_max, where=np.logical_and(detection_array == 0, rain_array > 0), facecolor='red',
                alpha=.5, label="Mis-Detection")

ax.fill_between(x, rain_max, where=np.logical_and(detection_array == 1, rain_array == 0), facecolor='blue',
                alpha=.5, label="False Alarm")
plt.legend()
plt.ylabel("Rain Rate [mm/hr]")
plt.xlabel("Sample Index")
plt.grid()
figure_name = f"Detections_{samples_type}.png"
plt.savefig(os.path.join(output_dir, figure_name))
plt.show(block=False)
plt.pause(5)
plt.close()

# Plot accumulated rainfall
plt.plot(np.cumsum(np.maximum(rain_hat_array, 0)), label="Two-Steps RNN")
plt.plot(np.cumsum(rain_ref_array), "--", label="Reference")
plt.grid()
plt.legend()
plt.ylabel("Accumulated Rain-Rate[mm]")
plt.xlabel("# Samples")
figure_name = f"Accumulated_Rain_Rate_{samples_type}.png"
plt.savefig(os.path.join(output_dir, figure_name))
plt.show(block=False)
plt.pause(5)
plt.close()

# Plot Rain Rate estimation
ref = rain_ref_array[start_idx:end_idx]
hat = rain_hat_array[start_idx:end_idx]
x = np.arange(start_idx, end_idx)

# Compute Pearson correlation - Avoid zero variance issue
if np.std(ref) == 0 or np.std(hat) == 0:
    corr = float('nan')  # or corr = 0.0 if you prefer
else:
    corr = np.corrcoef(ref, hat)[0, 1]

plt.figure()
plt.plot(x, ref, label="Reference", linestyle="--")
plt.plot(x, hat, label="Estimated")
plt.xlabel("Sample Index")
plt.ylabel(f"Rain Rate [mm/15 min]")
plt.title(f"Predicted vs. Reference Rain Rate\nSamples {start_idx}–{end_idx}, Corr = {corr:.2f}")
plt.grid()
plt.legend()
plt.tight_layout()
figure_name = f"RainRate_{samples_type}.png"
plt.savefig(os.path.join(output_dir, figure_name))
plt.show(block=False)
plt.pause(5)
plt.close()
