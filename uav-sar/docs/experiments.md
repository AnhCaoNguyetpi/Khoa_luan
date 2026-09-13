# Thiết kế Thực nghiệm và Ánh xạ 5 Tiers

Bản phân tích kết quả đã thực thi, log kiểm thử và các giới hạn kết luận được trình bày trong [Phân tích kết quả](results_analysis.md).

Hệ thống thực nghiệm được chuẩn hóa thành 5 lớp (Tiers 1–5) tương ứng với Bảng "Năm lớp thực nghiệm phục vụ kiểm chứng đa chiều" trong đề cương bài báo hội nghị.

Toàn bộ các thí nghiệm được khởi chạy thống nhất qua entry-point:

```bash
# Chạy toàn bộ 5 tiers:
python scripts/run_experiments.py --tier all

# Chạy kiểm thử nhanh (smoke-test):
python scripts/run_experiments.py --tier all --quick

# Chạy một tier cụ thể (ví dụ Tier 2 kiểm chứng evaluator):
python scripts/run_experiments.py --tier 2
```

Kết quả được ghi tự động vào `results/tier_{1..5}_results.json` và bảng tổng hợp `results/all_tiers_summary.json`.

---

## Chi tiết 5 Tiers Thực nghiệm

### Tier 1 — Phân tích Cơ chế trên Lưới $3 \times 3$
* **File mã nguồn:** `src/sar_uav/experiments/tier1_mechanism.py`
* **Mục tiêu:** Minh họa trực quan sự thay đổi quyết định lộ trình và dwell khi thay đổi bất định trong 3 tình huống:
  1. *Scenario 1:* Tập giả thuyết ensemble đem lại lợi ích rõ rệt so với nominal.
  2. *Scenario 2:* Mô hình danh định nominal chuẩn xác ($S=1$), ensemble và nominal trùng khớp.
  3. *Scenario 3:* Bất định lớn nhưng phân bố vị trí ban đầu quá tập trung nên quyết định tối ưu không đổi.

### Tier 2 — Độ chính xác Thuật toán và Đo lường Tăng tốc Evaluator
* **File mã nguồn:** `src/sar_uav/experiments/tier2_evaluator_accuracy.py`
* **Mục tiêu:**
  - Kiểm chứng tính tương đương số học giữa `ForwardBackwardFastEvaluator` và `FullEvaluator` (sai số tuyệt đối $< 10^{-12}$).
  - Đo đạc thời gian tính toán trung bình (ms/eval) và mức tăng tốc so với `FullEvaluator` và `PrefixOnlyEvaluator`.
  - Đo lường độ lệch nghiệm tối ưu $\Delta_{\mathrm{opt}} = J^\star - J_{\mathrm{alg}}$ so với nghiệm chuẩn của `ExactBruteForcePlanner`.

### Tier 3 — Benchmark So sánh Đa phương pháp (Main Benchmark)
* **File mã nguồn:** `src/sar_uav/experiments/tier3_main_benchmark.py` & `src/sar_uav/experiments/tier3_multi_instance.py`
* **Cấu trúc hai cấp thực nghiệm tách rời (Decoupled two-level experimental hierarchy):**
  1. **Cấp Scenario & Chế độ Ground Truth:** Khảo sát trên 10 kịch bản/scenario độc lập với sự đa dạng về Prior $b_0$ (đa đỉnh, dải sườn núi, thung lũng dốc, bimodal, phân tán trung tâm), ma trận Markov khuếch tán và trôi dạt có hướng ($M$) và môi trường thảm phủ. Mỗi scenario được đánh giá đầy đủ qua cả **3 chế độ Ground Truth**:
     - *In-Ensemble (10 scenarios):* Môi trường thực bám theo một giả thuyết cụ thể trong tập ensemble của planner nhưng khác nominal.
     - *Nominal-Matched (10 scenarios):* Môi trường thực trùng khớp chính xác với nominal (đo lường chi phí bảo thủ của ensemble khi nominal chuẩn xác).
     - *Misspecified Out-of-Ensemble (10 scenarios):* Môi trường thực bị suy giảm cảm biến thực địa ngoài tập giả thuyết (sương mù thung lũng, trôi dạt xiết, che khuất tán rừng vượt ngưỡng).
     Tạo thành $10 \times 3 = 30$ evaluations hoàn chỉnh.
  2. **Cấp Mission trong Scenario:** 100 mission ngẫu nhiên mô phỏng đường đi mục tiêu và sự kiện cảm biến theo Common Random Numbers (CRN), tổng cộng **3.000 missions** đánh giá ghép cặp.
* **Mục tiêu so sánh đối đầu:**
  - **Nhóm 1 (Đóng góp thuật toán - Multi-start, Single-start & Evaluators):**
    - `Proposed_Fast`: Multi-start (2 starts: greedy ensemble + greedy nominal) + Forward-Backward Fast Evaluator (300+300 evals).
    - `SingleStart_Fast_FixedLS`: Single-start từ greedy ensemble với ngân sách Local Search cố định (300 evals).
    - `SingleStart_Fast_MatchedTotal`: Single-start bù trừ ngân sách evaluation ($B_{\mathrm{LS}}^{\mathrm{matched}} = 600 + N_{\mathrm{eval}}^{\mathrm{greedy\_nom}}$ evals, trần tổng evaluation $N_{\mathrm{total}}$ bằng đúng Multi-start).
    - `Proposed_PrefixOnly`: Multi-start + Prefix-only Evaluator (300+300 evals, bóc tách giá trị của backward continuation).
    - **4 cấu hình Toán tử Ablation (thu thập thống kê single-solve trên 10 scenario runs):**
      - `Proposed_Fast` (Đầy đủ: Replace=Có, Rebalance=Có)
      - `Ablation_NoReplace` (Bỏ Replace, giữ Rebalance)
      - `Ablation_NoRebalance` (Giữ Replace, bỏ Rebalance)
      - `Ablation_NoReplaceRebalance` (Bỏ cả Replace và Rebalance)
    - `Greedy_Lookahead`: Chiến lược tham lam phân bổ tức thời (Search SAR benchmark).
    - `GA_Baseline` (alias `Adaptive_GA` trong mã nguồn): Thuật toán di truyền chuẩn (Standard Genetic Algorithm baseline) triển khai qua `SimpleEvolutionaryPlanner` với chọn lọc giải đấu (tournament size 2), lai ghép 1 điểm (1-point crossover), đột biến ô và dwell ($p_{\mathrm{mut}} = 0.3$), kèm repair cắt đuôi bảo đảm an toàn/pin.
  - **Nhóm 2 (Đóng góp mô hình - Ensemble vs Nominal):**
    - `Nominal_2Start`: Đối chứng mô hình thuần túy (Pure Model baseline) sử dụng cùng giải thuật Multi-start 2-start nhưng tối ưu hóa trên mô hình nominal duy nhất ($S=1$), bóc tách chính xác ưu thế của ensemble khi cùng solver.
    - `Nominal_Planning`: Đối chứng pipeline 1-start trên mô hình nominal ($S=1$).
* **Chỉ số đánh giá & Thống kê suy luận:**
  - Tỷ lệ tìm thấy $DSR$, thời gian tìm trung bình giới hạn $RMST$, năng lượng tiêu thụ thực tế ($kJ$), Calibration Gap ($J_{\mathcal{S}} - DSR$).
  - Khoảng tin cậy Scenario Cluster Bootstrap 95% và Student's $t$ CI với $df = N_{\mathrm{scenarios}} - 1$ kèm clipping miền vật lý.
  - Phân tích chi tiết theo từng chế độ Ground Truth (In-Ensemble, Nominal-Matched, Misspecified).
* **Nguyên tắc phân tích thực nghiệm:**
  - Báo cáo trung thực cả trường hợp phương pháp đề xuất vượt trội và trường hợp các biến thể ablation/single-start đạt kết quả tương đương.
  - Phân biệt rõ ràng giữa các tương phản có ý nghĩa thống kê ($p < 0.05$) và các khác biệt không có ý nghĩa thống kê (khoảng tin cậy chứa 0).

### Tier 4 — Kiểm tra Khả năng Chống chịu Sai đặc tả (Model Misspecification)
* **File mã nguồn:** `src/sar_uav/experiments/tier4_misspecification.py`
* **Mục tiêu:** Đánh giá độ bền vững khi mô hình vật lý thực $q^\star \notin \mathcal{S}$ (ngoài tập giả thuyết của planner) dưới các mức độ suy giảm cảm biến và che phủ thực địa $\alpha \in [0.3, 2.0]$. Xác định ranh giới mà tại đó ensemble bắt đầu suy giảm lợi ích.

### Tier 5 — Khảo sát Khả năng Mở rộng (Scalability & Quality-vs-Runtime)
* **File mã nguồn:** `src/sar_uav/experiments/tier5_scalability.py`
* **Mục tiêu:** Đo đạc chất lượng nghiệm và thời gian hội tụ khi mở rộng quy mô không gian lưới ($|G| \in \{16, 36, 64, 100\}$) và số lượng phi đội UAV ($m \in \{1, 2, 4\}$).

---

## Kỹ thuật Ghép cặp Ngẫu nhiên (Common Random Numbers)
* **File mã nguồn:** `src/sar_uav/experiments/rng.py`
* Tuân thủ nghiêm ngặt protocol ghép cặp thông qua lớp `IndexedRNGStream` lập chỉ mục 4 phần tử:
  $$(\text{mission\_idx}, \text{tick\_t}, \text{uav\_k}, \text{event\_type})$$
  bảo đảm hai UAV khác nhau quan sát cùng thời điểm không dùng chung số ngẫu nhiên, giữ vững tính độc lập có điều kiện theo không gian (A1).
