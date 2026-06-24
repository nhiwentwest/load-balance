# Load Balance: MARL Cloud Resource Management

Dự án này là một hệ thống cân bằng tải và cấp phát tài nguyên (VM Allocation) trên nền tảng điện toán đám mây. Dự án sử dụng phương pháp **Học tăng cường đa tác tử (Multi-Agent Reinforcement Learning - MARL)** kết hợp với **Kiến trúc Lambda** (Online + Offline learning) thông qua các mô hình nhận diện dự đoán như Autoformer và BiLSTM. 

Hệ thống được thiết kế phân tán: Môi trường mô phỏng (Environment) chạy bằng **Java (CloudSim Plus)** và các mô hình Học sâu (Agents) chạy bằng **Python (PyTorch)**, giao tiếp qua cầu nối Py4J.

## Cấu trúc Project

### 1. Phần Python (`python/`) - Mô hình & Huấn luyện
Thư mục này chứa toàn bộ logic về trí tuệ nhân tạo, RL agents và lập lịch.
- **`marl_v4_train.py`**: File chạy huấn luyện chính. Sử dụng thuật toán CTDE PPO (Centralized Training Decentralized Execution) với 4 agents song song.
- **`models.py`**: Chứa định nghĩa kiến trúc mạng nơ-ron (PyTorch) của các tác tử (Actor) và Centralized Critic.
- **`autoformer_detector.py`**: Mô hình Autoformer dùng để dự đoán quá tải (Overload) của Host trong tương lai.
- **`lstm_underload_detector.py`**: Mô hình BiLSTM dùng để nhận diện trạng thái thấp tải (Underload).
- **`lambda_runner.py`**: Trình chạy chính thức theo kiến trúc Lambda (chạy dự đoán Online và thu thập dữ liệu).
- **`offline_trainer.py`**: Lớp chạy ngầm (Background thread) để tự động huấn luyện lại (retrain) Autoformer và LSTM dựa trên dữ liệu lỗi (Event Database).
- **`scheduler.py`**: Bộ lập lịch lai (Hybrid Scheduler) kết hợp kết quả của RL Agents và các luật cứng.
- **`cloudsim_gym_env.py`**: Môi trường OpenAI Gym wrapper giúp đóng gói dữ liệu từ Java CloudSim để truyền cho Python.
- **`config.py`**: Các siêu tham số (Hyperparameters), đường dẫn và cấu hình chung.

### 2. Phần Java (`src/main/java/com/dacn/`) - Mô phỏng CloudSim
Thư mục này chứa môi trường giả lập Data Center vật lý và các thuật toán cấp phát cơ sở (Baselines).
- **`advanced/Py4jBridge.java`**: Điểm neo kết nối (Gateway) để CloudSim lắng nghe lệnh cấp phát từ Python.
- **`advanced/UtilizationModelGenParallel.java`** & **`UtilizationModelAzure.java`**: Các model sinh dữ liệu CPU Usage mô phỏng theo hàm hoặc theo dữ liệu thật từ Azure.
- **`VmAllocationPolicyACO.java`, `VmAllocationPolicyPSO.java`, `VmAllocationPolicyTabuSearch.java`**: Các thuật toán cấp phát truyền thống (Thuật toán Kiến, Bầy chim, Tabu Search) dùng làm Baseline để chạy so sánh.
- **`CompareAllAlgorithms.java`** & **`Benchmark.java`**: Các script chạy so sánh hiệu năng của mô hình MARL với các thuật toán truyền thống.

## Cách làm việc với Mô hình

1. **Huấn luyện mô hình từ đầu (Train MARL)**:
   - Đảm bảo môi trường Java (`Py4jBridge`) đã được chạy trước để mở cổng giao tiếp.
   - Chạy script `python python/marl_v4_train.py` để bắt đầu huấn luyện. Trọng số sẽ được lưu dưới dạng file `.pt`.

2. **Chạy hệ thống thời gian thực (Lambda Architecture)**:
   - Chạy `python python/lambda_runner.py` để chạy hệ thống kết hợp dự đoán trực tuyến và tái huấn luyện định kỳ.

3. **Chạy đánh giá (Benchmark/Evaluate)**:
   - Để chạy kiểm tra 1 episode: dùng script `python/eval_v7_demo.py`.
   - Để chạy so sánh với các thuật toán Heuristic: chạy các class `CompareAll*.java` bên phía Java.
