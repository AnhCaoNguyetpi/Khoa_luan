# uav-sar — Multi-UAV Routing and Search-Effort Allocation under Persistent Detection-Model Uncertainty

Framework mô phỏng và tối ưu hóa phối hợp nhiều UAV cho bài toán tìm kiếm cứu nạn (Search and Rescue - SAR) trong điều kiện sai số mô hình cảm biến kéo dài suốt nhiệm vụ.

Framework được xây dựng bám sát các nguyên lý mô hình hóa và thuật toán trong đề tài nghiên cứu:
> **Multi-UAV Routing and Search-Effort Allocation under Persistent Detection-Model Uncertainty**  
> *(Tác giả: Cao Nguyệt Ánh — Đề tài luận văn tốt nghiệp & bài báo hội nghị)*

---

## 1. Điểm Nổi bật và Đóng góp Thuật toán

1. **Formulation Phối hợp 4 Thành phần:** Tối ưu hóa đồng thời phân công UAV, lộ trình ghé thăm (hỗ trợ revisit), thời lượng tìm kiếm (dwell time) và thời gian chờ (active wait) dưới tập giả thuyết sai số cố định $\mathcal{S} = \{q^s, w_s\}$.
2. **Khối lượng Xác suất 2 Chiều (Forward Mass & Backward Continuation):**
   - Tiền tố Forward Unnormalized Mass: $u_t^-(i, s), u_t^+(i, s)$.
   - Hậu tố Backward Continuation: $V_t(i, s)$ với điều kiện biên $V_H(i, s) = 1.0$.
   - Hàm mục tiêu xác suất tích lũy kỳ vọng: $J_{\mathcal{S}}(a) = 1 - \sum_{i, s} u_{H-1}^+(i, s)$.
3. **Bộ Đánh giá Nghiệm 3 Cấp độ (3-Tier Evaluator):**
   - `FullEvaluator`: Tính toán lại toàn bộ từ $t=0 \to H-1$.
   - `PrefixOnlyEvaluator`: Tái sử dụng tiền tố $u_{t_a}^-$, lan truyền tiếp tới $H-1$.
   - `ForwardBackwardFastEvaluator`: Tái sử dụng tiền tố $u_{t_a}^-$, chỉ lan truyền trong đoạn biến đổi $[t_a, t_b]$ và ghép nối với $V_{t_b+1}$.
4. **Không gian Láng giềng 8 Toán tử (Chống bẫy kẹt ngân sách & Lazy Generation):**
   - `Replace Visit`: Thay thế vùng tìm kiếm bằng vùng khác trong một bước duy nhất (chống kẹt khi pin/thời gian đã bão hòa).
   - `Dwell Rebalance`: Tái phân bổ thời lượng giữa các lần ghé, bảo toàn tổng dwell ticks.
   - `Insert Visit` (hỗ trợ revisit), `Delete Visit`, `Swap / 2-opt`, `Relocate`, `Change Dwell`, `Adjust Wait`.
   - Điều khiển thứ tự toán tử: `fixed`, `round_robin`, `shuffled`.
5. **Ràng buộc Động học & An toàn Quay về:**
   - Hệ thống đẳng thức nối tiếp hành động: $s_{k1} = w_{k1} + \bar\tau_{o, v_{k1}}^k$ và $s_{k,\ell+1} = s_{k\ell} + d_{k\ell} + \bar\tau_{v_{k\ell}, v_{k,\ell+1}}^k + w_{k,\ell+1}$.
   - Kiểm tra an toàn trước từng chặng bay dở dang: $B_k(t) \ge \bar e_{uv}^k + \bar e_{vo}^k + R_k$.
   - Quy tắc quay về khi có phát hiện: UAV tại trạm về depot; UAV đang bay hoàn thành chặng tới $v$ rồi quay về depot theo đường bay khả thi ngắn nhất. Năng lượng sortie được đồng bộ tất định với ConstraintChecker.
6. **Benchmark Đa Kịch bản Tách bạch 3 Ground Truth (Decoupled Benchmark):**
   - Đánh giá 10 kịch bản địa hình/thời tiết độc lập qua cả 3 chế độ Ground Truth: *In-Ensemble*, *Nominal-Matched*, và *Misspecified* (tổng cộng 30 đơn vị đánh giá đối đầu $\times$ 100 mission = 3.000 mission/phương pháp).
   - Đối chứng thuần mô hình `Nominal_2Start` (cùng 2-start budget, cùng solver, cùng 2 điểm khởi tạo greedy).
   - Ước lượng độ bất định cấp kịch bản: *Stratified Scenario Cluster Bootstrap CI (95%)* và *Scenario-Aggregated Student's $t$ CI*.

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
│   ├── experiments.md           # Đặc tả chi tiết 5 Tiers thực nghiệm
│   └── results_analysis.md      # Báo cáo phân tích kết quả benchmark chuẩn tắc
├── scripts/
│   ├── run_mission.py           # Chạy một demo mission với Local Search + Simulation
│   ├── run_experiments.py       # Entry-point thực thi 5 Tiers thực nghiệm
│   ├── run_multi_instance.py    # Chạy benchmark đa kịch bản tách bạch 3 Ground Truth
│   ├── plot_paper_figures.py    # Vẽ Biểu đồ 1, 2, 3 (Mechanism, Evaluator, Case Study)
│   ├── plot_multi_instance_figures.py # Vẽ Biểu đồ 4, 5 (Multi-Instance & Operator Ablation)
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
│   │   ├── neighborhood.py      # 8 toán tử biến đổi láng giềng (fixed, round_robin, shuffled)
│   │   ├── local_search.py      # Algorithm 1: Joint Route-Effort Local Search (Single & Multi-start)
│   │   └── baselines.py         # Greedy Lookahead, Adaptive GA, Exact Brute Force
│   ├── sim/
│   │   └── mission.py           # ScheduleSimulator với return-to-depot protocol
│   └── experiments/
│       ├── rng.py               # IndexedRNGStream 4 phần tử cho CRN
│       ├── stats.py             # Scenario Cluster Bootstrap, Student's t, McNemar test
│       ├── tier1_mechanism.py   # Tier 1: Phân tích cơ chế Jensen lưới 3x3
│       ├── tier2_evaluator_accuracy.py  # Tier 2: Độ chính xác & speedup evaluator
│       ├── tier3_main_benchmark.py      # Tier 3: Benchmark cơ sở
│       ├── tier3_multi_instance.py      # Tier 3: Benchmark đa kịch bản 3 Ground Truth
│       ├── tier4_misspecification.py    # Tier 4: Kiểm tra sai đặc tả q* not in S
│       └── tier5_scalability.py         # Tier 5: Khảo sát khả năng mở rộng Quality-vs-Runtime
├── tests/                       # 99 unit tests tự động (100% pass)
└── results/                     # Kết quả JSON, logs XML và biểu đồ PDF/PNG
    ├── logs/validation_latest.xml # Log thực thi test suite
    ├── figures/                 # 5 biểu đồ nghiên cứu chuẩn IEEE/ACM
    └── multi_instance_tier3_results.json # Dữ liệu thực nghiệm chính thức
```

---

## 3. Cài đặt & Kiểm thử

```bash
# 1. Kích hoạt môi trường ảo (trên Windows)
.venv\Scripts\activate

# 2. Chạy toàn bộ 99 unit tests kiểm thử tự động
pytest -q --junitxml=results/logs/validation_latest.xml
```

---

## 4. Hướng dẫn Thực thi Thực nghiệm (Quickstart)

```bash
# 1. Chạy một mission thử nghiệm với Fast Evaluator
python scripts/run_mission.py --evaluator fast --uavs 2 --horizon 15

# 2. Chạy Tier 1 (Phân tích cơ chế Jensen trên lưới 3x3)
python scripts/run_experiments.py --tier 1

# 3. Chạy Tier 2 (Đo lường độ chính xác và tốc độ Fast vs Full vs Prefix-only)
python scripts/run_experiments.py --tier 2

# 4. Chạy toàn bộ 5 Tiers thực nghiệm
python scripts/run_experiments.py --tier all

# 5. Chạy Benchmark đa kịch bản tách bạch 3 Ground Truth (Tier 3 Multi-Instance)
python scripts/run_multi_instance.py --num-instances 10 --num-missions 100 --operator-order fixed

# 6. Xuất toàn bộ biểu đồ báo cáo khoa học (Figures 1-5)
python scripts/plot_paper_figures.py
python scripts/plot_multi_instance_figures.py
```

---

## 5. Tóm tắt Kết quả Thực nghiệm Chính thức

Dữ liệu tổng hợp trên 10 kịch bản $\times$ 3 Ground Truth regimes = 30 lượt đánh giá đối đầu (3.000 mission/phương pháp):

| Phương pháp | DSR (%) | CI 95% Bootstrap Scenario | RMST (tick) | Năng lượng (kJ) | $J$ Ground Truth |
|---|---:|:---:|---:|---:|---:|
| **Proposed Fast (2-start, Ens)** | **27,87** | [22,70%; 33,30%] | **12,522** | 280,28 | **0,27249** |
| **Nominal 2-Start (Pure Model)** | 26,87 | [21,70%; 32,37%] | 12,628 | 282,40 | 0,27072 |
| **Single-start FixedLS** | 28,03 | [23,00%; 33,43%] | 12,541 | 279,33 | 0,26350 |
| **Single-start MatchedTotal** | 28,20 | [23,20%; 33,57%] | 12,545 | 279,41 | 0,26355 |
| **Nominal Planning (1-start)** | 27,10 | [22,00%; 32,53%] | 12,592 | 281,66 | 0,26830 |
| **Greedy Lookahead** | 26,10 | [21,00%; 31,57%] | 12,688 | 275,17 | 0,25044 |
| **Adaptive GA** | 26,03 | [21,77%; 30,53%] | 12,677 | 272,13 | 0,25789 |

- **So sánh thuần mô hình (`Proposed_Fast` vs `Nominal_2Start`):** $\Delta\text{DSR} = \mathbf{+1,00\%}$ (CI 95%: [+0,23%; +1,70%], $p_{\text{boot}} = 0,0144$, Wilcoxon $p = 0,0371$), cứu nạn nhanh hơn **0,105 tick**.
- **Tính phòng thủ khi sai đặc tả (*Misspecified*):** Proposed đạt DSR **25,60%** vs Nominal_2Start **23,90%** ($\Delta\text{DSR} = \mathbf{+1,70\%}$).
- Chi tiết đầy đủ xem tại [docs/results_analysis.md](docs/results_analysis.md).
