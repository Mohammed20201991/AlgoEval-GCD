# -*- coding: utf-8 -*-
"""

<img src="https://upload.wikimedia.org/wikipedia/commons/c/cd/Savitzky_golay_local_regression_wl045_pd04.gif" alt="Savitzky-Golay Filter Illustration">
"""

# Example for MultiWindowSavitzkyGolayFilter vis
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class MultiWindowSavitzkyGolayFilter:
    """
    Multi-Window Savitzky-Golay Filter for time series preprocessing.
    Supports both 1D and 2D time series (multi-feature).
    """
    def __init__(self, window_sizes=[15, 31, 61], polyorder=2):
        self.window_sizes = window_sizes
        self.polyorder = polyorder

    def apply_filter(self, data):
        data = np.asarray(data)

        is_1d = False
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            is_1d = True

        filtered_data = np.zeros_like(data)

        for window in self.window_sizes:
            if window % 2 == 0:
                window += 1
            if window > data.shape[0]:
                window = data.shape[0] if data.shape[0] % 2 != 0 else data.shape[0] - 1

            for i in range(data.shape[1]):
                filtered = savgol_filter(data[:, i], window_length=window, polyorder=self.polyorder)
                filtered_data[:, i] += filtered / len(self.window_sizes)

        return filtered_data.squeeze() if is_1d else filtered_data


# Create noisy 2D signal (3 features over 100 timesteps)
np.random.seed(0)
t = np.linspace(0, 4*np.pi, 100)
signals = np.stack([
    np.sin(t) + 0.2 * np.random.randn(100),
    np.cos(t) + 0.2 * np.random.randn(100),
    np.sin(2*t) + 0.2 * np.random.randn(100)
], axis=1)

# Apply filter
filter = MultiWindowSavitzkyGolayFilter()
filtered_signals = filter.apply_filter(signals)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
titles = ['Signal 1 (sin)', 'Signal 2 (cos)', 'Signal 3 (sin 2t)']

for i in range(3):
    axs[i].plot(t, signals[:, i], label='Noisy', alpha=0.5, linestyle='--')
    axs[i].plot(t, filtered_signals[:, i], label='Filtered', linewidth=2)
    axs[i].set_title(titles[i])
    axs[i].legend()
    axs[i].grid(True)

plt.xlabel("Time")
plt.tight_layout()
plt.show()

"""![Savitzky-Golay Animation](https://upload.wikimedia.org/wikipedia/commons/8/8d/Savitzky_Golay_Animation_Wl_65_Po_9_w_Poly_Boundary_Fit.gif)

# 0- Mount Drive
"""

from google.colab import drive
drive.mount('/content/drive')

import torch
print(torch.cuda.is_available())

"""# 1- Load Dataset"""

import os
dir_path = "/content/drive/MyDrive/AI-GoogleCluster"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Created directory: {dir_path}")
else:
    print(f"Directory already exists: {dir_path}")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/AI-GoogleCluster

"""## We do one time"""

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("derrickmwiti/google-2019-cluster-sample")

# print("Path to dataset files:", path)

# Save copy to my Drive
# !cp -r /root/.cache/kagglehub/datasets/derrickmwiti/google-2019-cluster-sample/versions/1/* /content/drive/MyDrive/AI-GoogleCluster/

file_path = '/content/drive/MyDrive/AI-GoogleCluster/borg_traces_data.csv'
# file_path = '/content/drive/MyDrive/AI-GoogleCluster/processed_borg_traces_data.csv'

import pandas as pd
import ast
from ast import literal_eval
import numpy as np
import ast
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import seaborn as sns

"""# Data Pre-processing

### Skip process if it is exist
"""

# Output file path
output_path = '/content/drive/MyDrive/AI-GoogleCluster/processed_borg_traces_data.csv'

# Check if processed file already exists
if os.path.exists(output_path):
    print(f"Processed file already exists at {output_path}. Skipping processing.")
else:
    def safe_literal_eval(x):
        if pd.isna(x) or x == '' or x == '[]' or x == '{}':
            return {}
        try:
            return literal_eval(str(x))
        except (ValueError, SyntaxError):
            return {}

    def process_constraint_column(x):
        if pd.isna(x) or x in ('', '[]', '""', '{}'):
            return {}
        try:
            if isinstance(x, str):
                parsed = literal_eval(x)
            else:
                parsed = x
            if isinstance(parsed, list):
                return parsed[0] if parsed and isinstance(parsed[0], dict) else {}
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}

    def process_cpu_distribution(x):
        if pd.isna(x) or x == '' or x == '[]':
            return np.nan
        try:
            if isinstance(x, str):
                cleaned = x.replace('[', '').replace(']', '').strip()
                if not cleaned:
                    return np.nan
                values = [float(v) for v in cleaned.split()]
                return np.mean(values) if values else np.nan
            elif isinstance(x, list):
                return np.mean(x)
            return np.nan
        except (ValueError, SyntaxError, TypeError):
            return np.nan

    dtype_spec = {
        'constraint': str,
        'start_after_collection_ids': str,
        'average_usage': str,
        'maximum_usage': str,
        'random_sample_usage': str,
        'cpu_usage_distribution': str,
        'tail_cpu_usage_distribution': str,
        'start_after_collection_ids': str,
    }

    df = pd.read_csv('/content/drive/MyDrive/AI-GoogleCluster/borg_traces_data.csv',
                     dtype=dtype_spec,
                     low_memory=False)

    nested_cols = ['resource_request', 'average_usage', 'maximum_usage', 'random_sample_usage']
    for col in nested_cols:
        if col in df.columns:
            expanded = df[col].apply(safe_literal_eval).apply(pd.Series)
            expanded = expanded.add_prefix(f'{col}_')
            df = pd.concat([df.drop(col, axis=1), expanded], axis=1)

    cpu_cols = {
        'cpu_usage_distribution': 'cpu_usage_distribution_average',
        'tail_cpu_usage_distribution': 'tail_cpu_usage_distribution_average',
        'start_after_collection_ids': 'start_after_collection_ids_average',
    }

    for col, new_col in cpu_cols.items():
        if col in df.columns:
            df[new_col] = df[col].apply(process_cpu_distribution)
            df = df.drop(col, axis=1)

    if 'constraint' in df.columns:
        constraint_data = df['constraint'].apply(process_constraint_column)
        constraint_expanded = pd.json_normalize(constraint_data).add_prefix('constraint_')
        df = df.drop('constraint', axis=1)
        df = pd.concat([df, constraint_expanded], axis=1)

    new_cols = [c for c in df.columns if any(nc in c for nc in nested_cols + ['constraint'] + list(cpu_cols.values()))]
    failed_index = df.columns.get_loc('failed') if 'failed' in df.columns else len(df.columns)
    old_cols = [c for c in df.columns if c not in new_cols]
    new_order = old_cols[:failed_index] + new_cols + old_cols[failed_index:]
    df = df[new_order]

    df.to_csv(output_path, index=False)

    print(f"Data processed and saved to {output_path}")
    print(f"Inserted new columns before 'failed': {new_cols}")

import gc
gc.collect()

output_path = '/content/drive/MyDrive/AI-GoogleCluster/processed_borg_traces_data.csv'
df = pd.read_csv(output_path)

# Show the shape of the dataset
print("Shape:", df.shape)

# Display the first 5 rows
df.head()

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Drop columns with high missing data
df.drop(['constraint_name', 'constraint_value','constraint_relation'], axis=1, inplace=True)

# Show column names
print("\nColumns:\n", df.columns)

# Get data types
print("\nData types:\n", df.dtypes)

df['collections_events_type'][0] , df['event'][0]

print(df['event'].unique())
print(df['event'].nunique())

"""Label Encoding (simple integer mapping)
no ordinal relationship
"""

df['event'] = df['event'].astype('category').cat.codes
print(df['event'].unique())

df['vertical_scaling'].unique() , df['scheduler'].unique()

# it might be needed here to do later
df['user'][:2]

# Drop columns with high missing data
df.drop(['user', 'collection_name','collection_logical_name','Unnamed: 0'], axis=1, inplace=True)

"""### Process Missing Data"""

# For Numerical we use Mean
df['cycles_per_instruction'].fillna(df['cycles_per_instruction'].mean(), inplace=True)
df['memory_accesses_per_instruction'].fillna(df['memory_accesses_per_instruction'].mean(), inplace=True)
df['resource_request_cpus'].fillna(df['resource_request_cpus'].mean(), inplace=True)
df['resource_request_memory'].fillna(df['resource_request_memory'].mean(), inplace=True)
df['maximum_usage_memory'].fillna(df['resource_request_memory'].mean(), inplace=True)
df['cpu_usage_distribution_average'].fillna(df['cpu_usage_distribution_average'].mean(), inplace=True)
df['tail_cpu_usage_distribution_average'].fillna(df['tail_cpu_usage_distribution_average'].mean(), inplace=True)
df.drop(['random_sample_usage_memory', 'start_after_collection_ids_average'], axis=1, inplace=True)

# For Catogerical we use mode
df['vertical_scaling'].fillna(df['vertical_scaling'].mode()[0], inplace=True)
df['scheduler'].fillna(df['scheduler'].mode()[0], inplace=True)

df['memory_accesses_per_instruction'][0]

df['cycles_per_instruction'][10]

# Get data types
print("\nData types:\n", df.dtypes)

# Get data types
print("\nData types:\n", df.dtypes)

# Basic statistical summary
print("\nStatistics:\n", df.describe())

sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Heatmap")
plt.show()

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Show the shape of the dataset
print("Shape:", df.shape)

# Display the first 5 rows
df.head()

# df.head(100).to_csv('/content/drive/MyDrive/AI-GoogleCluster/sample_processed_100_rows.csv', index=False)
# df.to_csv('/content/drive/MyDrive/AI-GoogleCluster/cleaned_processed_all_rows.csv', index=False)
df.head(1000).to_csv('/content/drive/MyDrive/AI-GoogleCluster/sample_processed_1000_rows.csv', index=False)

# Sample size per column (adjust to available memory)
SAMPLE_SIZE = 100000

# Max number of plots
MAX_PLOTS = 20

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Optional: filter out columns with enough valid values
numeric_cols = [col for col in numeric_cols if df[col].notnull().sum() > 100]

# Limit to first N numeric columns
numeric_cols = numeric_cols[:MAX_PLOTS]

# Plot with sampling
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    data = df[col].dropna()
    if len(data) > SAMPLE_SIZE:
        data = data.sample(SAMPLE_SIZE, random_state=42)
    sns.histplot(data, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import gc
gc.collect()

import gc
gc.collect()

import gc
gc.collect()

class MultiWindowSavitzkyGolayFilter:
    """
    Multi-Window Savitzky-Golay Filter for time series preprocessing
    """
    def __init__(self, window_sizes=[15, 31, 61], polyorder=2):
        self.window_sizes = window_sizes
        self.polyorder    = polyorder

    def apply_filter(self, data):
        """
        Apply multi-window SG filter and combine results
        """
        filtered_data = np.zeros_like(data)

        for window in self.window_sizes:
            if window % 2 == 0:
                window += 1  # ensure odd window size
            if window > len(data):
                window = len(data) if len(data) % 2 != 0 else len(data)-1

            filtered = savgol_filter(data, window, self.polyorder)
            filtered_data += filtered / len(self.window_sizes)

        return filtered_data

class SymmetryAwareAttention(nn.Module):
    """
    Symmetry-Aware Attention Mechanism for Spiking Neural Networks
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key   = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.symmetry_weights = nn.Parameter(torch.randn(hidden_dim))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Symmetry-aware attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))

        # Add symmetry bias
        reversed_K = torch.flip(K, dims=[1])
        symmetry_scores = torch.matmul(Q, reversed_K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        combined_scores = attention_scores + self.gamma * symmetry_scores

        attention_weights = torch.softmax(combined_scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with proper dimension handling
    """
    def __init__(self, threshold=1.0, decay=0.9, hidden_dim=64):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.hidden_dim = hidden_dim
        self.membrane_potential = None

    def reset(self):
        self.membrane_potential = None

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        if self.membrane_potential is None:
            # Initialize with correct dimensions (batch_size, hidden_dim)
            self.membrane_potential = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        # Update membrane potential (sum over sequence dimension)
        input_current = x.sum(dim=1)  # Sum over sequence dimension
        self.membrane_potential = self.decay * self.membrane_potential + input_current
        spike = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spike)

        # Return spikes with shape (batch_size, 1, hidden_dim)
        return spike.unsqueeze(1)

class MASNN(nn.Module):
    """
    Multi-Dimensional Attention Spiking Neural Network with fixed dimensions
    """
    def __init__(self, input_dim=31, hidden_dim=64, output_dim=1, time_steps=10):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Encoder layers
        self.encoder   = nn.Linear(input_dim, hidden_dim)
        self.attention = SymmetryAwareAttention(hidden_dim, hidden_dim)

        # Spiking layers with explicit hidden_dim
        self.lif1 = LIFNeuron(hidden_dim=hidden_dim)
        self.lif2 = LIFNeuron(hidden_dim=hidden_dim)

        # Decoder layers
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)

        # Initialize spike trains
        spike_train = torch.zeros(batch_size, self.time_steps, self.hidden_dim, device=x.device)

        # Reset neuron states for new batch
        self.lif1.reset()
        self.lif2.reset()

        # Temporal processing
        for t in range(self.time_steps):
            # Encode input
            encoded = self.encoder(x)  # (batch_size, seq_len, hidden_dim)

            # Apply attention
            attended = self.attention(encoded)  # (batch_size, seq_len, hidden_dim)

            # Spiking layers
            spikes1 = self.lif1(attended)  # (batch_size, 1, hidden_dim)
            spikes2 = self.lif2(spikes1)   # (batch_size, 1, hidden_dim)

            # Store spikes for this timestep
            spike_train[:, t, :] = spikes2.squeeze(1)

        # Decode output (average over time steps)
        output = self.decoder(spike_train.mean(dim=1))  # (batch_size, output_dim)

        return output

class SecretaryBirdOptimizer:
    """
    Secretary Bird Optimization Algorithm for hyperparameter tuning
    """
    def __init__(self, population_size=20, max_iter=100, search_space=None):
        self.population_size = population_size
        self.max_iter     = max_iter
        self.search_space = search_space
        self.best_solution= None
        self.best_fitness = float('inf')

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = {}
            for param, bounds in self.search_space.items():
                if isinstance(bounds[0], int):
                    solution[param] = np.random.randint(bounds[0], bounds[1])
                else:
                    solution[param] = np.random.uniform(bounds[0], bounds[1])
            population.append(solution)
        return population

    def hunting_behavior(self, solution, best_solution):
        new_solution = solution.copy()
        for param in self.search_space.keys():
            step = np.random.uniform(0, 1) * (best_solution[param] - solution[param])
            new_solution[param] += step

            # Ensure within bounds
            if isinstance(self.search_space[param][0], int):
                new_solution[param] = int(np.clip(new_solution[param],
                                                self.search_space[param][0],
                                                self.search_space[param][1]-1))
            else:
                new_solution[param] = np.clip(new_solution[param],
                                             self.search_space[param][0],
                                             self.search_space[param][1])
        return new_solution

    def stomping_behavior(self, solution):
        new_solution = solution.copy()
        for param in self.search_space.keys():
            if np.random.rand() < 0.3:  # 30% chance to stomp (random walk)
                if isinstance(self.search_space[param][0], int):
                    new_solution[param] = np.random.randint(self.search_space[param][0],
                                                           self.search_space[param][1])
                else:
                    new_solution[param] = np.random.uniform(self.search_space[param][0],
                                                          self.search_space[param][1])
        return new_solution

    def optimize(self, model, train_loader, val_loader, criterion):
        population = self.initialize_population()

        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                # Set model parameters based on current solution
                self.set_model_params(model, population[i])

                # Evaluate fitness
                fitness = self.evaluate_fitness(model, train_loader, val_loader, criterion)

                # Update personal best
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = population[i].copy()

            # Update population
            for i in range(self.population_size):
                # Hunting phase
                population[i] = self.hunting_behavior(population[i], self.best_solution)

                # Stomping phase
                if np.random.rand() < 0.5:
                    population[i] = self.stomping_behavior(population[i])

        return self.best_solution

    def set_model_params(self, model, params):
        # Set model hyperparameters based on optimization solution
        for name, param in params.items():
            if hasattr(model, name):
                setattr(model, name, param)
            elif name in model.attention.__dict__:
                setattr(model.attention, name, param)
            elif name in model.lif1.__dict__:
                setattr(model.lif1, name, param)
                setattr(model.lif2, name, param)

    def evaluate_fitness(self, model, train_loader, val_loader, criterion):
        # Train for a few epochs
        optimizer = optim.Adam(model.parameters())
        model.train()
        for epoch in range(3):  # Short training for fitness evaluation
            for x, y in train_loader:
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                val_loss += criterion(output, y).item()

        return val_loss / len(val_loader)

class MASNN_WL_RTSP_CS:
    def __init__(self, input_dim=None, hidden_dim=64, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scaler = MinMaxScaler()
        self.filter = MultiWindowSavitzkyGolayFilter()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_data(self, data_path):
        # Load and preprocess Google cluster trace data
        data = pd.read_csv(data_path)

        # Select relevant features and target
        # 'Unnamed: 0',
        features = ['time', 'instance_events_type', 'collection_id', 'scheduling_class', 'collection_type','priority','alloc_collection_id','instance_index',
                    'machine_id','collections_events_type','vertical_scaling','scheduler','start_time','end_time','assigned_memory','page_cache_memory',
                    'cycles_per_instruction','memory_accesses_per_instruction','sample_rate','cluster','event','resource_request_cpus','resource_request_memory','average_usage_cpus',
                    'average_usage_memory','maximum_usage_cpus','maximum_usage_memory','random_sample_usage_cpus', 'cpu_usage_distribution_average',
                    'tail_cpu_usage_distribution_average' ]


        target = ['failed']

        X = data[features].values
        y = data[target].values

        # Apply filtering and scaling
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_filtered[:, i] = self.filter.apply_filter(X[:, i])

        X_scaled = self.scaler.fit_transform(X_filtered)
        y_scaled = self.scaler.fit_transform(y)

        self.input_dim = X_scaled.shape[1]  # Should be 31
        self.model = MASNN(input_dim=self.input_dim,
                         hidden_dim=self.hidden_dim,
                         output_dim=self.output_dim).to(self.device)

        return X_scaled, y_scaled

    def create_sequences(self, X, y, seq_length=10):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        # Define optimization search space
        search_space = {
            'threshold': (0.5, 2.0),
            'decay': (0.7, 0.99),
            'gamma': (-1.0, 1.0),
            'time_steps': (5, 20)
        }

        # Optimize hyperparameters
        optimizer   = SecretaryBirdOptimizer(search_space=search_space)
        best_params = optimizer.optimize(self.model, train_loader, val_loader, nn.MSELoss())

        # Set best parameters
        optimizer.set_model_params(self.model, best_params)

        # Full training with best parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    output = self.model(x)
                    val_loss += criterion(output, y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

    def evaluate(self, X_test, y_test):
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_t).cpu().numpy()

        # Inverse transform predictions and actual values
        predictions = self.scaler.inverse_transform(predictions)
        y_test = self.scaler.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(predictions)))

        return mse, mae, rmsle, predictions

# Example usage
if __name__ == "__main__":
    # Initialize system without input_dim (will be set during preprocessing)
    system = MASNN_WL_RTSP_CS()

    # Use sampled data path
    X, y = system.preprocess_data("/content/drive/MyDrive/AI-GoogleCluster/sample_processed_100_rows.csv")

    # Create sequences
    X_seq, y_seq = system.create_sequences(X, y)

    # Split data
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))

    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]

    print(f"Input shape: {X_train.shape}")  # Should be (samples, seq_len, features)
    print(f"Expected features: {system.model.input_dim}, Actual: {X_train.shape[2]}")
    assert X_train.shape[2] == system.model.input_dim, "Feature dimension mismatch!"

    # Train the model
    system.train(X_train, y_train, X_val, y_val)

    # Evaluate
    mse, mae, rmsle, predictions = system.evaluate(X_test, y_test)
    print(f"Test Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSLE: {rmsle:.4f}")

# Example usage
if __name__ == "__main__":
    # Initialize system without input_dim (will be set during preprocessing)
    system = MASNN_WL_RTSP_CS()

    # Use sampled data path
    X, y = system.preprocess_data("/content/drive/MyDrive/AI-GoogleCluster/cleaned_processed_all_rows.csv")

    # Create sequences
    X_seq, y_seq = system.create_sequences(X, y)

    # Split data
    train_size = int(0.7 * len(X_seq))
    val_size   = int(0.15 * len(X_seq))

    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]

    print(f"Input shape: {X_train.shape}")  # Should be (samples, seq_len, features)
    print(f"Expected features: {system.model.input_dim}, Actual: {X_train.shape[2]}")
    assert X_train.shape[2] == system.model.input_dim, "Feature dimension mismatch!"

    # Train the model
    system.train(X_train, y_train, X_val, y_val)

    # Evaluate
    mse, mae, rmsle, predictions = system.evaluate(X_test, y_test)
    print(f"Test Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSLE: {rmsle:.4f}")

# Example usage
if __name__ == "__main__":
    # Initialize system without input_dim (will be set during preprocessing)
    system = MASNN_WL_RTSP_CS()

    # Use sampled data path
    X, y = system.preprocess_data("/content/drive/MyDrive/AI-GoogleCluster/cleaned_processed_all_rows.csv")

    # Create sequences
    X_seq, y_seq = system.create_sequences(X, y)

    # Split data
    train_size = int(0.7 * len(X_seq))
    val_size   = int(0.15 * len(X_seq))

    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]

    print(f"Input shape: {X_train.shape}")  # Should be (samples, seq_len, features)
    print(f"Expected features: {system.model.input_dim}, Actual: {X_train.shape[2]}")
    assert X_train.shape[2] == system.model.input_dim, "Feature dimension mismatch!"

    # Train the model
    system.train(X_train, y_train, X_val, y_val)

    # Evaluate
    mse, mae, rmsle, predictions = system.evaluate(X_test, y_test)
    print(f"Test Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSLE: {rmsle:.4f}")

# cleaned_processed_all_rows.csv
# Example usage
if __name__ == "__main__":
    # Initialize system without input_dim (will be set during preprocessing)
    system = MASNN_WL_RTSP_CS()

    # Use sampled data path
    X, y = system.preprocess_data("/content/drive/MyDrive/AI+Cloud/cleaned_processed_all_rows.csv")

    # Create sequences
    X_seq, y_seq = system.create_sequences(X, y)

    # Split data
    train_size = int(0.7 * len(X_seq))
    val_size   = int(0.15 * len(X_seq))

    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]

    print(f"Input shape: {X_train.shape}")  # Should be (samples, seq_len, features)
    print(f"Expected features: {system.model.input_dim}, Actual: {X_train.shape[2]}")
    assert X_train.shape[2] == system.model.input_dim, "Feature dimension mismatch!"

    # Train the model
    system.train(X_train, y_train, X_val, y_val)

    # Evaluate
    mse, mae, rmsle, predictions = system.evaluate(X_test, y_test)
    print(f"Test Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSLE: {rmsle:.4f}")

from tqdm import trange, tqdm

def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    self.model.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    val_labels = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch_X, batch_y in loop:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_outputs = self.model(val_tensor).squeeze()
        val_loss = criterion(val_outputs, val_labels).item()

        print(f"Epoch {epoch}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

"""# 2- Second Expierment
Key improvements made:
1- Enhanced Architecture:

- Increased hidden_dim from 64 to 128

- Added second attention layer

- Added dropout (0.2) and ReLU activation

- Deeper decoder network

2- Training Improvements:

- Increased epochs from 5 to 50

- Increased batch_size from 32 to 64

- Added learning rate scheduling with ReduceLROnPlateau

- Better learning rate (0.001) and Adam optimizer

3- Stability Improvements:

- More robust LIF neuron implementation

- Better weight initialization

- Improved sequence handling

4- Monitoring:

- Added learning rate to epoch printout

- Better model saving/loading


"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

class MultiWindowSavitzkyGolayFilter:
    def __init__(self, window_sizes=[15, 31, 61], polyorder=2):
        self.window_sizes = window_sizes
        self.polyorder = polyorder

    def apply_filter(self, data):
        filtered_data = np.zeros_like(data)
        for window in self.window_sizes:
            if window % 2 == 0:
                window += 1
            if window > len(data):
                window = len(data) if len(data) % 2 != 0 else len(data)-1
            filtered = savgol_filter(data, window, self.polyorder)
            filtered_data += filtered / len(self.window_sizes)
        return filtered_data

class SymmetryAwareAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.symmetry_weights = nn.Parameter(torch.randn(hidden_dim))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        reversed_K = torch.flip(K, dims=[1])
        symmetry_scores = torch.matmul(Q, reversed_K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        combined_scores = attention_scores + self.gamma * symmetry_scores

        attention_weights = torch.softmax(combined_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9, hidden_dim=128):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.hidden_dim = hidden_dim
        self.membrane_potential = None

    def reset(self):
        self.membrane_potential = None

    def forward(self, x):
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        input_current = x.sum(dim=1)
        self.membrane_potential = self.decay * self.membrane_potential + input_current
        spike = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spike)
        return spike.unsqueeze(1)

class MASNN(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=128, output_dim=1, time_steps=20):
        super().__init__()
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Enhanced architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.attention1 = SymmetryAwareAttention(hidden_dim, hidden_dim)
        self.attention2 = SymmetryAwareAttention(hidden_dim, hidden_dim)

        self.lif1 = LIFNeuron(hidden_dim=hidden_dim)
        self.lif2 = LIFNeuron(hidden_dim=hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        spike_train = torch.zeros(batch_size, self.time_steps, self.hidden_dim, device=x.device)

        self.lif1.reset()
        self.lif2.reset()

        for t in range(self.time_steps):
            encoded = self.encoder(x)
            attended1 = self.attention1(encoded)
            attended2 = self.attention2(attended1)

            spikes1 = self.lif1(attended2)
            spikes2 = self.lif2(spikes1)

            spike_train[:, t, :] = spikes2.squeeze(1)

        output = self.decoder(spike_train.mean(dim=1))
        return output

class MASNN_WL_RTSP_CS:
    def __init__(self, input_dim=None, hidden_dim=128, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scaler = MinMaxScaler()
        self.filter = MultiWindowSavitzkyGolayFilter()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_data(self, data_path):
        data = pd.read_csv(data_path)
        features = [ 'time', 'instance_events_type', 'collection_id',
                   'scheduling_class', 'collection_type', 'priority', 'alloc_collection_id',
                   'instance_index', 'machine_id', 'collections_events_type',
                   'vertical_scaling', 'scheduler', 'start_time', 'end_time',
                   'assigned_memory', 'page_cache_memory', 'cycles_per_instruction',
                   'memory_accesses_per_instruction', 'sample_rate', 'cluster', 'event',
                   'resource_request_cpus', 'resource_request_memory', 'average_usage_cpus',
                   'average_usage_memory', 'maximum_usage_cpus', 'maximum_usage_memory',
                   'random_sample_usage_cpus', 'cpu_usage_distribution_average',
                   'tail_cpu_usage_distribution_average']
        target = ['failed']

        X = data[features].values
        y = data[target].values

        X_filtered = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_filtered[:, i] = self.filter.apply_filter(X[:, i])

        X_scaled = self.scaler.fit_transform(X_filtered)
        y_scaled = self.scaler.fit_transform(y)

        self.input_dim = X_scaled.shape[1]
        self.model = MASNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(self.device)

        return X_scaled, y_scaled

    def create_sequences(self, X, y, seq_length=10):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )

        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    output = self.model(x)
                    val_loss += criterion(output, y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

    def evaluate(self, X_test, y_test):
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_t).cpu().numpy()

        predictions = self.scaler.inverse_transform(predictions)
        y_test = self.scaler.inverse_transform(y_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(predictions)))

        return mse, mae, rmsle, predictions

if __name__ == "__main__":
    system = MASNN_WL_RTSP_CS()
    X, y = system.preprocess_data("/content/drive/MyDrive/AI-GoogleCluster/sample_processed_100_rows.csv")

    X_seq, y_seq = system.create_sequences(X, y)

    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))

    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]

    print(f"Input shape: {X_train.shape}")
    system.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)

    mse, mae, rmsle, predictions = system.evaluate(X_test, y_test)
    print(f"Test Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSLE: {rmsle:.4f}")

"""**Results Analysis**
1- Training Progress:

- Excellent initial convergence (Train/Val loss dropped from 0.26/0.20 to 0.19/0.18 in first 15 epochs)

- Learning rate scheduling worked effectively (5 reductions from 0.001 to 0.000008)

- Final validation loss: 0.1798 (35% better than previous 0.27)

2- Test Performance:

- MSE improved by 10.2% (0.3257 vs 0.3625)

- MAE improved by 5.3% (0.5199 vs 0.5491)

- RMSLE improved by 8.3% (0.3749 vs 0.4088)

"""

import gc
gc.collect()

class MultiWindowSavitzkyGolayFilter:
    """
    Multi-Window Savitzky-Golay Filter for time series preprocessing
    """
    def __init__(self, window_sizes=[15, 31, 61], polyorder=2):
        self.window_sizes = window_sizes
        self.polyorder    = polyorder

    def apply_filter(self, data):
        """
        Apply multi-window SG filter and combine results
        """
        filtered_data = np.zeros_like(data)

        for window in self.window_sizes:
            if window % 2 == 0:
                window += 1  # ensure odd window size
            if window > len(data):
                window = len(data) if len(data) % 2 != 0 else len(data)-1

            filtered = savgol_filter(data, window, self.polyorder)
            filtered_data += filtered / len(self.window_sizes)

        return filtered_data

# Enhanced LIF Neuron with membrane decay
class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.95, hidden_dim=128):  # Increased decay from 0.9 to 0.95
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.hidden_dim = hidden_dim
        self.membrane_potential = None
        # NEW: Learnable decay parameter
        self.decay_param = nn.Parameter(torch.tensor([self.decay]))

    def reset(self):
        self.membrane_potential = None

    def forward(self, x):
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        # NEW: Use learnable decay with sigmoid constraint
        effective_decay = torch.sigmoid(self.decay_param)
        input_current = x.sum(dim=1)
        self.membrane_potential = effective_decay * self.membrane_potential + input_current
        spike = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spike)
        return spike.unsqueeze(1)

class SymmetryAwareAttention(nn.Module):
    """
    Symmetry-Aware Attention Mechanism for Spiking Neural Networks
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key   = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.symmetry_weights = nn.Parameter(torch.randn(hidden_dim))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Symmetry-aware attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))

        # Add symmetry bias
        reversed_K = torch.flip(K, dims=[1])
        symmetry_scores = torch.matmul(Q, reversed_K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        combined_scores = attention_scores + self.gamma * symmetry_scores

        attention_weights = torch.softmax(combined_scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output

# NEW: Enhanced MASNN Architecture
class MASNN(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=256, output_dim=1, time_steps=30):  # Increased dimensions
        super().__init__()
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # NEW: Enhanced encoder with skip connection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # NEW: Layer normalization
            nn.LeakyReLU(0.1),        # NEW: LeakyReLU
            nn.Dropout(0.3)            # Increased dropout
        )
        self.skip = nn.Linear(input_dim, hidden_dim)  # NEW: Skip connection

        # NEW: Dual attention mechanism
        self.attention1 = SymmetryAwareAttention(hidden_dim, hidden_dim)
        self.attention2 = SymmetryAwareAttention(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(0.2)  # NEW: Attention dropout

        self.lif1 = LIFNeuron(hidden_dim=hidden_dim)
        self.lif2 = LIFNeuron(hidden_dim=hidden_dim)

        # NEW: Deeper decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//4, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        spike_train = torch.zeros(batch_size, self.time_steps, self.hidden_dim, device=x.device)

        self.lif1.reset()
        self.lif2.reset()

        # NEW: Skip connection implementation
        encoded = self.encoder(x) + self.skip(x.mean(dim=1, keepdim=True))  # Mean reduces sequence dim

        for t in range(self.time_steps):
            attended1 = self.attention1(encoded)
            attended2 = self.attn_dropout(self.attention2(attended1))  # NEW: Attention dropout

            spikes1 = self.lif1(attended2)
            spikes2 = self.lif2(spikes1)

            spike_train[:, t, :] = spikes2.squeeze(1)

        output = self.decoder(spike_train.mean(dim=1))
        return output

# NEW: Enhanced Training Class
class MASNN_WL_RTSP_CS:
    def __init__(self, input_dim=None, hidden_dim=256, output_dim=1):  # Increased hidden_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scaler = MinMaxScaler()
        self.filter = MultiWindowSavitzkyGolayFilter()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128):  # Increased epochs/bs
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 pin_memory=True)  # NEW: Faster data loading
        val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True)

        criterion = nn.MSELoss()
        # NEW: AdamW optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(),
                               lr=0.001,
                               weight_decay=0.01)
        # NEW: Enhanced scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=7,  # Increased patience
            factor=0.5,
            min_lr=1e-6
        )

        best_val_loss = float('inf')
        best_epoch = 0
        no_improvement = 0  # NEW: Early stopping counter

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)  # NEW: Faster zero_grad
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                # NEW: Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    output = self.model(x)
                    val_loss += criterion(output, y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # NEW: Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improvement = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improvement += 1
                if no_improvement >= 10:  # Patience of 10
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # NEW: Enhanced evaluation with TTA
    def evaluate(self, X_test, y_test, n_augments=3):
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            # Base prediction
            output = self.model(X_test_t)
            all_preds.append(output.cpu().numpy())

            # Test-time augmentation
            for _ in range(n_augments-1):
                noise = torch.randn_like(X_test_t) * 0.01  # Small noise
                pred = self.model(X_test_t + noise)
                all_preds.append(pred.cpu().numpy())

        predictions = np.mean(all_preds, axis=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_test = self.scaler.inverse_transform(y_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(predictions)))

        return mse, mae, rmsle, predictions

if __name__ == "__main__":
    system = MASNN_WL_RTSP_CS()
    X, y = system.preprocess_data("/content/drive/MyDrive/AI-GoogleCluster/sample_processed_100_rows.csv")

    X_seq, y_seq = system.create_sequences(X, y)

    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))

    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]

    print(f"Input shape: {X_train.shape}")
    system.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)

    mse, mae, rmsle, predictions = system.evaluate(X_test, y_test)
    print(f"Test Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSLE: {rmsle:.4f}")