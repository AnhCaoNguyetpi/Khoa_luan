# Thiết kế Thực nghiệm và Ánh xạ 5 Tiers

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
* **File mã nguồn:** `src/sar_uav/experiments/tier3_main_benchmark.py`
* **Mục tiêu:** So sánh đối đầu giữa các chiến lược đối chứng theo Bảng 2 của đề cương:
  - **Nhóm 1 (Đóng góp thuật toán - Multi-start & Evaluators):**
    - `Proposed_Fast`: Multi-start (2 starts: greedy ensemble + greedy nominal) + Forward-Backward Fast Evaluator (300+300 evals).
    - `SingleStart_Fast_FixedLS`: Single-start từ greedy ensemble với ngân sách Local Search cố định (600 evals).
    - `SingleStart_Fast_MatchedTotal`: Single-start bù trừ ngân sách evaluation:
      - Ngân sách Local Search: $B_{\mathrm{LS}}^{\mathrm{matched}} = B_{\mathrm{LS}}^{\mathrm{total}} + N_{\mathrm{eval}}^{\mathrm{greedy\_nom}} = 600 + 228 = 828$ evals.
      - Tổng evaluation toàn phương pháp: $N_{\mathrm{total}} = N_{\mathrm{eval}}^{\mathrm{greedy\_ens}} + B_{\mathrm{LS}}^{\mathrm{matched}} = 190 + 828 = 1018$ evals.
      > *Lưu ý về protocol:* Đối chứng này đảm bảo tổng số lần truy vấn hàm mục tiêu $N_{\mathrm{total}}$ của Single-start bằng đúng $N_{\mathrm{total}}$ của Multi-start ($N_{\mathrm{eval}}^{\mathrm{greedy\_ens}} + N_{\mathrm{eval}}^{\mathrm{greedy\_nom}} + B_{\mathrm{LS}}^{\mathrm{start1}} + B_{\mathrm{LS}}^{\mathrm{start2}} = 190 + 228 + 300 + 300 = 1018$). Đây là đối chứng kiểm soát **trần số lượt đánh giá hàm mục tiêu** (matched evaluation budget ceiling), không phải cùng tổng thời gian thực (wall-clock time), do chi phí tính toán mỗi bước đánh giá khác nhau giữa pha greedy và các toán tử lân cận.
    - `Proposed_PrefixOnly`: Multi-start + Prefix-only Evaluator (300+300 evals, bóc tách giá trị của backward continuation).
    - `Ablation_NoReplaceRebalance`: Multi-start bỏ 2 toán tử Replace và Dwell Rebalance (khảo sát tác động của kích thước không gian lân cận).
    - `Greedy_Lookahead`: Chiến lược tham lam phân bổ tức thời (Search SAR benchmark).
    - `GA_Baseline` (alias `Adaptive_GA` trong mã nguồn): Thuật toán di truyền chuẩn (Standard Genetic Algorithm baseline) triển khai qua `SimpleEvolutionaryPlanner` với chọn lọc giải đấu (tournament size 2), lai ghép 1 điểm (1-point crossover), đột biến ô và dwell với xác suất cố định ($p_{\mathrm{mut}} = 0.3$), và cơ chế repair cắt đuôi bảo đảm ràng buộc an toàn/pin. Không sử dụng cơ chế tự thích nghi tham số.
  - **Nhóm 2 (Đóng góp mô hình):**
    - `Nominal_Planning`: Lập kế hoạch dưới mô hình điểm ($S=1$) rồi đánh giá trên môi trường thực.
* **Chỉ số đánh giá:** Tỷ lệ tìm thấy $DSR$, thời gian tìm trung bình giới hạn $RMST$, năng lượng tiêu thụ thực tế ($kJ$), và Calibration Gap ($J_{\mathcal{S}} - D_r$).
* **Nguyên tắc phân tích thực nghiệm:**
  - Báo cáo trung thực cả trường hợp phương pháp đề xuất vượt trội và trường hợp các biến thể ablation/nominal đạt kết quả tốt hơn.
  - Phân tích nguyên nhân thuật toán: chi phí ngân sách của từng toán tử, tính trơn của search landscape dưới mô hình đơn vs mô hình ensemble.
  - Mọi khác biệt về DSR trên tập mẫu nhỏ (ví dụ 15 mission, mỗi lần phát hiện ứng với 6.67%) phải đi kèm khoảng tin cậy và không suy diễn vượt quá dữ liệu.

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
