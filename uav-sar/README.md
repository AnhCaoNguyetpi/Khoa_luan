# uav-sar — Multi-UAV Routing and Search-Effort Allocation under Persistent Detection-Model Uncertainty

Framework mô phỏng và tối ưu hóa phối hợp nhiều UAV cho bài toán tìm kiếm cứu nạn (Search and Rescue - SAR) trong điều kiện sai số mô hình cảm biến kéo dài suốt nhiệm vụ.

Framework được xây dựng bám sát các nguyên lý mô hình hóa và thuật toán trong đề tài nghiên cứu:
> **Multi-UAV Routing and Search-Effort Allocation under Persistent Detection-Model Uncertainty**  
> *(Tác giả: Cao Nguyệt Ánh — Đề tài luận văn & bài báo hội nghị)*

---

## 1. Điểm nổi bật và Đóng góp Thuật toán

1. **Formulation Phối hợp 4 Thành phần:** Tối ưu hóa đồng thời phân công UAV, lộ trình ghé thăm (hỗ trợ revisit), thời lượng tìm kiếm (dwell time) và thời gian chờ (active wait) dưới tập giả thuyết sai số cố định $\mathcal{S} = \{q^s, w_s\}$.
2. **Khối lượng Xác suất 2 Chiều (Forward Mass & Backward Continuation):**
   - Tiền tố Forward Unnormalized Mass: $u_t^-(i, s), u_t^+(i, s)$.
   - Hậu tố Backward Continuation: $V_t(i, s)$ với điều kiện biên $V_H(i, s) = 1.0$.
   - Hàm mục tiêu xác suất tích lũy kỳ vọng: $J_{\mathcal{S}}(a) = 1 - \sum_{i, s} u_{H-1}^+(i, s)$.
3. **Bộ Đánh giá Nghiệm 3 Cấp độ (3-Tier Evaluator):**
   - `FullEvaluator`: Tính toán lại toàn bộ từ $t=0 \to H-1$.
   - `PrefixOnlyEvaluator`: Tái sử dụng tiền tố $u_{t_a}^-$, lan truyền tiếp tới $H-1$.
   - `ForwardBackwardFastEvaluator`: Tái sử dụng tiền tố $u_{t_a}^-$, chỉ lan truyền trong $[t_a, t_b]$ và ghép với $V_{t_b+1}$.
4. **Không gian Láng giềng 8 Toán tử (Chống bẫy kẹt ngân sách):**
   - `Replace Visit`: Thay thế vùng tìm kiếm bằng vùng khác trong một bước duy nhất (chống kẹt khi pin/thời gian đã bão hòa).
   - `Dwell Rebalance`: Tái phân bổ thời lượng giữa các lần ghé, bảo toàn tổng dwell ticks.
   - `Insert Visit` (hỗ trợ revisit), `Delete Visit`, `Swap / 2-opt`, `Relocate`, `Change Dwell`, `Adjust Wait`.
5. **Ràng buộc Động học & An toàn Quay về:**
   - Hệ thống đẳng thức nối tiếp hành động $s_{k1} = w_{k1} + \bar\tau_{o, v_{k1}}^k$ và $s_{k,\ell+1} = s_{k\ell} + d_{k\ell} + \bar\tau_{v_{k\ell}, v_{k,\ell+1}}^k + w_{k,\ell+1}$.
   - Kiểm tra an toàn trước từng chặng bay dở dang: $B_k(t) \ge \bar e_{uv}^k + \bar e_{vo}^k + R_k$.
   - Quy tắc quay về khi có phát hiện: UAV tại trạm về depot; UAV đang bay hoàn thành chặng tới $v$ rồi quay về depot theo đường bay khả thi ngắn nhất. Năng lượng sortie được đồng bộ tất định với ConstraintChecker.

---

## 2. Cấu trúc Thư mục Dự án

```
uav-sar/
├── configs/                     # Cấu hình nhiệm vụ (default_mission, real_mission)
├── data/
│   └── areas/tay_nguyen_real/   # Dữ liệu GIS Tây Nguyên (DEM GLO-30, WorldCover, OSM)
├── docs/
│   ├── proposal/proposal.tex    # Bản thảo đề cương bài báo hội nghị (LaTeX)
│   ├── architecture.md          # Sơ đồ kiến trúc & luồng dữ liệu tối ưu hóa
│   └── experiments.md           # Đặc tả chi tiết 5 Tiers thực nghiệm
├── scripts/
│   ├── run_mission.py           # Chạy một demo mission với Local Search + Simulation
│   ├── run_experiments.py       # Entry-point thực thi 5 Tiers thực nghiệm
│   ├── plot_paper_figures.py    # Vẽ 3 biểu đồ nghiên cứu từ dữ liệu thực nghiệm
│   └── build_real_area.py       # Xây dựng lưới GIS Tây Nguyên từ dữ liệu nguồn
├── src/sar_uav/
│   ├── detection/
│   │   └── hypothesis.py        # Tập giả thuyết S, mô hình exponential exposure
│   ├── belief/
│   │   ├── joint_mass.py        # Động cơ JointMassEngine (Forward Mass & Backward Continuation)
│   │   └── motion.py            # Ma trận chuyển động Markov M(i,j)
│   ├── planning/
│   │   ├── constraints.py       # Ràng buộc đẳng thức, pin dự phòng & pre-leg check
│   │   ├── evaluator.py         # 3 cấp độ: Full, Prefix-only, Forward-Backward Fast Evaluator
│   │   ├── neighborhood.py      # 8 toán tử biến đổi láng giềng lazy generator
│   │   ├── local_search.py      # Algorithm 1: Joint Route-Effort Local Search
│   │   └── baselines.py         # Greedy Lookahead, SimpleEvolutionaryPlanner (Adaptive GA), Exact Brute Force
│   ├── sim/
│   │   └── mission.py           # ScheduleSimulator với return-to-depot protocol
│   └── experiments/
│       ├── rng.py               # IndexedRNGStream 4 phần tử cho CRN
│       ├── stats.py             # Wilson confidence intervals, McNemar test, bootstrap
│       ├── tier1_mechanism.py   # Tier 1: Phân tích cơ chế lưới 3x3
│       ├── tier2_evaluator_accuracy.py  # Tier 2: Độ chính xác & speedup evaluator
│       ├── tier3_main_benchmark.py      # Tier 3: Benchmark so sánh đối đầu các baselines
│       ├── tier4_misspecification.py    # Tier 4: Kiểm tra sai đặc tả q* not in S
│       └── tier5_scalability.py         # Tier 5: Khảo sát khả năng mở rộng Quality-vs-Runtime
├── tests/                       # 62 unit tests kiểm chứng chặt chẽ
└── results/                     # Kết quả JSON, CSV và biểu đồ thực nghiệm
```

---

## 3. Cài đặt & Kiểm thử

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate          # Trên Windows

# Chạy toàn bộ 62 unit tests
pytest
```

---

## 4. Hướng dẫn Sử dụng (Quickstart)

```bash
# 1. Chạy một mission thử nghiệm với Fast Evaluator
python scripts/run_mission.py --evaluator fast --uavs 2 --horizon 15

# 2. Chạy Tier 1 (Phân tích cơ chế trên lưới 3x3)
python scripts/run_experiments.py --tier 1

# 3. Chạy Tier 2 (Đo lường độ chính xác và tốc độ Fast vs Full vs Prefix-only)
python scripts/run_experiments.py --tier 2

# 4. Chạy toàn bộ 5 Tiers thực nghiệm
python scripts/run_experiments.py --tier all

# 5. Vẽ biểu đồ bài báo từ dữ liệu thực nghiệm thật (lưu vào results/figures/)
python scripts/plot_paper_figures.py
```
