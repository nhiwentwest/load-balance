import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: [Batch, Seq_len, Channels]
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AutoCorrelationLayer(nn.Module):
    """
    Simplified AutoCorrelation mechanism tailored for 1D CPU Utilization
    """
    def __init__(self, d_model):
        super(AutoCorrelationLayer, self).__init__()
        self.d_model = d_model

    def forward(self, queries, keys, values):
        B, L, E = queries.shape
        _, S, _ = keys.shape
        
        # padding
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :]
            keys = keys[:, :L, :]

        # FFT
        q_fft = torch.fft.rfft(queries.permute(0, 2, 1), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 1), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)
        
        # Aggregate based on correlation weights
        weights = F.softmax(corr, dim=-1)
        out = torch.einsum('b e l, b e l -> b l e', weights, values.permute(0, 2, 1))
        return out.contiguous()

class AutoformerDetector(nn.Module):
    """
    Autoformer model for Proactive PM State Detection
    Input: [Batch, seq_len] (Historical CPU)
    Output: [Batch, pred_len] (Predicted future CPU)
    """
    def __init__(self, seq_len=20, pred_len=5, d_model=32):
        super(AutoformerDetector, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.decomp = series_decomp(kernel_size=5)
        self.value_embedding = nn.Linear(1, d_model)
        
        self.autocorr = AutoCorrelationLayer(d_model)
        
        self.trend_proj = nn.Linear(seq_len, pred_len)
        self.seasonal_proj = nn.Linear(d_model * seq_len, pred_len)
        
    def forward(self, x):
        # x: [Batch, seq_len]
        x = x.unsqueeze(-1) # [Batch, seq_len, 1]
        
        # 1. Decomposition
        seasonal_init, trend_init = self.decomp(x)
        
        # 2. Trend projection
        trend_part = self.trend_proj(trend_init.squeeze(-1)) # [Batch, pred_len]
        
        # 3. Seasonal auto-correlation
        x_emb = self.value_embedding(seasonal_init) # [Batch, seq_len, d_model]
        seasonal_corr = self.autocorr(x_emb, x_emb, x_emb) # [Batch, seq_len, d_model]
        
        # 4. Flatten and project seasonal part
        seasonal_part = self.seasonal_proj(seasonal_corr.view(x.size(0), -1)) # [Batch, pred_len]
        
        # 5. Final prediction
        prediction = trend_part + seasonal_part
        return torch.clamp(prediction, 0.0, 1.0) # CPU utilization is between 0 and 1

def detect_pm_state(model, history_cpu, underload_threshold=None, overload_threshold=None):
    """
    Analyze the history and autoformer predictions to detect proactive PM state.
    """
    if underload_threshold is None or overload_threshold is None:
        from config import Config
        underload_threshold = Config.UNDERLOAD_THRESHOLD
        overload_threshold = Config.OVERLOAD_THRESHOLD
    model.eval()
    with torch.no_grad():
        x = torch.tensor([history_cpu], dtype=torch.float32)
        preds = model(x).squeeze(0).numpy()
        
    max_pred = max(preds)
    
    # Analyze recent history trend (last 5 intervals out of 20)
    recent_hist = history_cpu[-5:]
    recent_max = max(recent_hist)
    recent_avg = sum(recent_hist) / len(recent_hist)
    
    # PROACTIVE UNDERLOAD:
    # Both predicted future and recent history must be steadily low
    if max_pred < underload_threshold and recent_max < underload_threshold:
        return 'UNDERLOAD'
        
    # PROACTIVE OVERLOAD:
    # Predicted to spike and recent trend is already dangerously high
    trend_floor = max(underload_threshold, overload_threshold - (overload_threshold - underload_threshold) / 3.0)
    if max_pred > overload_threshold and recent_avg > trend_floor:
        return 'OVERLOAD'
        
    return 'NORMAL'

# Quick local test
if __name__ == '__main__':
    model = AutoformerDetector()
    # Mock CPU history of a PM that is doing fine, then suddenly spikes
    mock_history = [0.3]*15 + [0.6, 0.7, 0.8, 0.85, 0.9]
    state = detect_pm_state(model, mock_history)
    print(f"Detected PM State: {state}")
