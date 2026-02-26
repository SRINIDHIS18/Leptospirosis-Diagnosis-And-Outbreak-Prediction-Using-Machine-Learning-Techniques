from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import io
import matplotlib.pyplot as plt

# ---- Initializations ----
app = Flask(__name__)
CORS(app)

# ---- Data Loading and Preprocessing ----
df = pd.read_csv('FINAL.csv')  # Your CSV must have columns: week, year, state, cases

df['month'] = df['week'].apply(lambda x: (x - 1) // 4 + 1)
df_monthly = df.groupby(['year', 'state', 'month'])['cases'].sum().reset_index()
years = range(df['year'].min(), df['year'].max() + 1)
months = range(1, 13)
states = df['state'].unique()

full_index = pd.MultiIndex.from_product([years, states, months], names=['year', 'state', 'month'])
df_monthly = df_monthly.set_index(['year', 'state', 'month']).reindex(full_index).fillna(0).reset_index()
df_monthly['cases'] = df_monthly['cases'].astype(float)

# ---- PyTorch Dataset and Model ----
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

def min_max_scale(data):
    minv, maxv = np.min(data), np.max(data)
    if maxv == minv:
        return data / (maxv + 1e-10), minv, maxv
    return (data - minv) / (maxv - minv + 1e-10), minv, maxv

def train_and_predict(series, time_step=4, epochs=100, lr=0.001, future_steps=36):
    if len(series) < time_step+1 or np.sum(series) == 0:
        mean_val = np.mean(series[series > 0]) if np.any(series > 0) else 1
        return np.ones(future_steps) * mean_val

    scaled, minv, maxv = min_max_scale(series)
    X, y = create_dataset(scaled, time_step)
    if len(X) < 1:
        mean_val = np.mean(series[series > 0]) if np.any(series > 0) else 1
        return np.ones(future_steps) * mean_val

    dataset = TimeSeriesDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    last_seq = scaled[-time_step:]

    with torch.no_grad():
        for _ in range(future_steps):
            inp = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred_scaled = model(inp).item()
            pred_val = max(0, pred_scaled * (maxv - minv + 1e-10) + minv)
            preds.append(pred_val)
            last_seq = np.append(last_seq[1:], pred_scaled)

    return np.array(preds)

# ---- Precompute statewise data ----
time_step = 4
forecast_months = 36  # 3 years

yearly_historical = {}
future_predictions = {}

for state in sorted(states):
    state_df = df_monthly[df_monthly['state'] == state].sort_values(['year', 'month'])
    series = state_df['cases'].values
    yearly_historical[state] = state_df.groupby('year')['cases'].sum().to_dict()

    preds = train_and_predict(series, time_step, epochs=100, lr=0.001, future_steps=forecast_months)
    yearly_pred = [sum(preds[i*12:(i+1)*12]) for i in range(3)]
    future_predictions[state] = [max(1, round(x)) for x in yearly_pred]

# ---- API Endpoints ----

@app.route('/')
def home():
    return "Leptospirosis API - POST /api/outbreak_predict and /api/prediction with {'state': <state>}"

@app.route('/api/outbreak_predict', methods=['POST'])
def outbreak_predict():
    data = request.get_json()
    if 'state' not in data:
        return jsonify({"error": "State required"}), 400
    state = data['state'].upper()
    if state not in states:
        return jsonify({"error": "State not found"}), 404
    hist = {str(y): int(c) for y, c in yearly_historical[state].items() if y in [2022, 2023, 2024, 2025]}
    return jsonify({"state": state, "historical_cases": hist})

@app.route('/api/prediction', methods=['POST'])
def prediction():
    data = request.get_json()
    if 'state' not in data:
        return jsonify({"error": "State required"}), 400
    state = data['state'].upper()
    if state not in states:
        return jsonify({"error": "State not found"}), 404
    pred_years = [2026, 2027, 2028]
    pred_cases = future_predictions[state]
    return jsonify({"state": state, "predicted_cases": {str(k): v for k, v in zip(pred_years, pred_cases)}})

@app.route('/api/outbreak_predict/graph', methods=['POST'])
def outbreak_predict_graph():
    data = request.get_json()
    if 'state' not in data:
        return jsonify({"error": "State required"}), 400
    state = data['state'].upper()
    if state not in states:
        return jsonify({"error": "State not found"}), 404

    # Combine historical and future data for plot 
    hist = {y: c for y, c in yearly_historical[state].items() if y in [2022, 2023, 2024, 2025]}
    hist_years = list(hist.keys())
    hist_cases = list(hist.values())
    pred_years = [2026, 2027, 2028]
    pred_cases = future_predictions[state]
    years = hist_years + pred_years
    cases = hist_cases + pred_cases

    plt.figure(figsize=(8,5))
    plt.plot(years, cases, marker='o', linestyle='-', color='green')
    plt.title(f'Leptospirosis Cases in {state} (2022-2028)')
    plt.xlabel('Year')
    plt.ylabel('Cases')
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
