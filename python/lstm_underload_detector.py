"""
Phase 3: BiLSTM Underload Detector integration module.
Provides:
  - HostCPUBuffer: per-host sliding window (anti-fail #3)
  - LSTMUnderloadDetector: inference wrapper using BiLSTM
  - HostShutdownCooldown: hysteresis to prevent ping-pong migration
"""
import torch
import torch.nn as nn
import numpy as np


class UnderloadBiLSTM(nn.Module):
    """BiLSTM architecture for underload prediction."""
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_out))
    
    def predict_prob(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class HostCPUBuffer:
    """
    Maintain sliding window CPU history for each host.
    Anti-fail #3: Without this buffer, LSTM receives stale/zero data.
    """
    def __init__(self, num_hosts, window=10):
        self.window = window
        self.num_hosts = num_hosts
        # Initialize with neutral values (0.5 = mid-range)
        self.buffers = {i: [0.5] * window for i in range(num_hosts)}
    
    def update_all(self, cpu_utils):
        """Update all hosts at once. cpu_utils: array of shape (num_hosts,)."""
        for h in range(min(len(cpu_utils), self.num_hosts)):
            self.buffers[h].append(float(cpu_utils[h]))
            if len(self.buffers[h]) > self.window:
                self.buffers[h].pop(0)
    
    def update(self, host_id, cpu_util):
        """Update single host."""
        self.buffers[host_id].append(float(cpu_util))
        if len(self.buffers[host_id]) > self.window:
            self.buffers[host_id].pop(0)
    
    def get_sequence(self, host_id):
        """Get CPU sequence for LSTM input."""
        return list(self.buffers[host_id])
    
    def get_all_sequences(self):
        """Get sequences for all hosts as numpy array."""
        return np.array([self.buffers[h] for h in range(self.num_hosts)])
    
    def reset(self):
        """Reset all buffers (call at episode start)."""
        self.buffers = {i: [0.5] * self.window for i in range(self.num_hosts)}
    
    def reset_host(self, host_id):
        """Reset single host (call after shutdown)."""
        self.buffers[host_id] = [0.0] * self.window


class HostShutdownCooldown:
    """
    Hysteresis timer to prevent ping-pong migration.
    After a host is evacuated, it can't be re-evacuated for cooldown_steps.
    """
    def __init__(self, cooldown_steps=20):
        self.cooldown = {}  # host_id -> remaining cooldown
        self.cooldown_steps = cooldown_steps
    
    def can_shutdown(self, host_id):
        return self.cooldown.get(host_id, 0) == 0
    
    def mark_shutdown(self, host_id):
        self.cooldown[host_id] = self.cooldown_steps
    
    def tick(self):
        """Call once per step to decrement cooldowns."""
        for k in list(self.cooldown.keys()):
            self.cooldown[k] = max(0, self.cooldown[k] - 1)
            if self.cooldown[k] == 0:
                del self.cooldown[k]
    
    def reset(self):
        self.cooldown.clear()


class LSTMUnderloadDetector:
    """
    Complete underload detection using LSTM + Buffer + Hysteresis.
    
    Usage:
        detector = LSTMUnderloadDetector("lstm_underload.pt", num_hosts=20)
        detector.reset()                    # at episode start
        detector.update(cpu_utils)          # at each step
        underload_hosts = detector.detect() # get underload host indices
    """
    def __init__(self, model_path, num_hosts=20, window=10, 
                 threshold=0.75, cooldown_steps=20):
        self.num_hosts = num_hosts
        self.threshold = threshold
        
        # Load BiLSTM model
        self.model = UnderloadBiLSTM()
        if model_path and torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        elif model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu',
                                                   weights_only=True))
        self.model.eval()
        
        # Buffer and hysteresis
        self.buffer = HostCPUBuffer(num_hosts, window)
        self.cooldown = HostShutdownCooldown(cooldown_steps)
        
        self._enabled = True
    
    def reset(self):
        """Call at episode start."""
        self.buffer.reset()
        self.cooldown.reset()
    
    def update(self, cpu_utils):
        """Update CPU history. Call ONCE per step with current utils."""
        self.buffer.update_all(cpu_utils)
        self.cooldown.tick()
    
    def detect(self):
        """
        Returns list of host indices predicted to be underloaded.
        Respects cooldown (no re-detection within cooldown window).
        """
        if not self._enabled:
            return []
        
        underload_hosts = []
        
        for h in range(self.num_hosts):
            if not self.cooldown.can_shutdown(h):
                continue
            
            seq = self.buffer.get_sequence(h)
            
            # Skip hosts that are clearly active (fast check)
            if seq[-1] > 0.50:
                continue
            
            # LSTM prediction
            x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)  # (1, 10, 1)
            prob = self.model.predict_prob(x).item()
            
            if prob > self.threshold:
                underload_hosts.append(h)
                self.cooldown.mark_shutdown(h)
        
        return underload_hosts
    
    def detect_with_probs(self):
        """Returns (host_indices, probabilities) for debugging."""
        hosts, probs = [], []
        for h in range(self.num_hosts):
            seq = self.buffer.get_sequence(h)
            if seq[-1] > 0.50:
                continue
            x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)
            prob = self.model.predict_prob(x).item()
            if prob > self.threshold and self.cooldown.can_shutdown(h):
                hosts.append(h)
                probs.append(prob)
        return hosts, probs
