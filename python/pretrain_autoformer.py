"""
Pre-train Autoformer on historical CPU utilization data.
This MUST run before marl_train.py so the detector has meaningful weights.
Uses self-supervised learning: predict next 5 steps from 20-step windows.
"""
import torch
import torch.nn as nn
import numpy as np
from autoformer_detector import AutoformerDetector

def generate_synthetic_traces(n_traces=500, seq_len=200):
    """Generate realistic CPU utilization traces for pre-training."""
    traces = []
    for _ in range(n_traces):
        t = np.linspace(0, 24*np.pi, seq_len)
        # Base pattern: diurnal cycle
        base = 0.3 + 0.2 * np.sin(t / (2*np.pi) * 0.5)
        # Add spikes (bursty workload)
        n_spikes = np.random.randint(3, 10)
        for _ in range(n_spikes):
            center = np.random.randint(0, seq_len)
            width = np.random.randint(5, 20)
            height = np.random.uniform(0.3, 0.6)
            spike = height * np.exp(-0.5 * ((np.arange(seq_len) - center) / width) ** 2)
            base += spike
        # Add noise
        base += np.random.normal(0, 0.03, seq_len)
        base = np.clip(base, 0.01, 0.99)
        traces.append(base)
    return np.array(traces, dtype=np.float32)

def create_windows(traces, input_len=20, pred_len=5):
    """Slide a window over traces to create (input, target) pairs."""
    X, Y = [], []
    for trace in traces:
        for i in range(len(trace) - input_len - pred_len + 1):
            X.append(trace[i:i+input_len])
            Y.append(trace[i+input_len:i+input_len+pred_len])
    return np.array(X), np.array(Y)

def pretrain():
    print("=" * 60)
    print("Pre-training Autoformer on synthetic CPU traces...")
    print("=" * 60)
    
    # Try to load real traces from Java bridge first
    try:
        from py4j.java_gateway import JavaGateway
        gateway = JavaGateway()
        bridge = gateway.entry_point
        # Collect real data by running simulation
        print("[Pretrain] Collecting real CPU traces from CloudSim...")
        bridge.reset()
        real_histories = []
        for step in range(50):
            hist = np.array(bridge.getHostHistory())  # [NUM_HOSTS, 20]
            real_histories.append(hist.copy())
            # Step with no actions to just advance time
            empty_hosts = gateway.new_array(gateway.jvm.int, 0)
            empty_place = gateway.new_array(gateway.jvm.int, 0)
            result = np.array(bridge.step(empty_hosts, 0, empty_place))
            done = result[bridge.getGlobalStateDim() + 1] > 0.5
            if done:
                break
        gateway.shutdown()
        
        # Convert to traces: each host's utilization over time
        if len(real_histories) > 10:
            real_traces = []
            num_hosts = real_histories[0].shape[0]
            for h in range(num_hosts):
                trace = [real_histories[t][h][-1] for t in range(len(real_histories))]
                if len(trace) >= 25:
                    real_traces.append(np.array(trace, dtype=np.float32))
            if real_traces:
                print(f"[Pretrain] Collected {len(real_traces)} real traces of length {len(real_traces[0])}")
    except Exception as e:
        print(f"[Pretrain] No Java bridge available, using synthetic only: {e}")
        real_traces = []

    # Generate synthetic traces
    synthetic = generate_synthetic_traces(n_traces=500, seq_len=200)
    print(f"[Pretrain] Generated {len(synthetic)} synthetic traces")
    
    # Combine
    all_traces = synthetic
    if real_traces and len(real_traces) > 0:
        # Pad real traces if needed
        for rt in real_traces:
            if len(rt) >= 25:
                padded = np.pad(rt, (0, max(0, 200 - len(rt))), mode='wrap')[:200]
                all_traces = np.vstack([all_traces, padded.reshape(1, -1)])
    
    X, Y = create_windows(all_traces, input_len=20, pred_len=5)
    print(f"[Pretrain] Training samples: {len(X)}")
    
    # Train
    model = AutoformerDetector(seq_len=20, pred_len=5, d_model=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "autoformer_pretrained.pt")
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50 | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
    
    print(f"[Pretrain] Done! Best loss: {best_loss:.6f}")
    print(f"[Pretrain] Saved: autoformer_pretrained.pt")
    
    # Quick sanity check
    model.eval()
    with torch.no_grad():
        sample = X_tensor[:1]
        pred = model(sample).numpy()[0]
        actual = Y_tensor[0].numpy()
        print(f"[Pretrain] Sample prediction: {pred}")
        print(f"[Pretrain] Sample actual:     {actual}")
        print(f"[Pretrain] MAE: {np.mean(np.abs(pred - actual)):.4f}")

if __name__ == "__main__":
    pretrain()
