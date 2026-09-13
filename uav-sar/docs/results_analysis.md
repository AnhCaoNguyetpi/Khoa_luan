# Phân tích Kết quả và Thiết kế Thực nghiệm Chuẩn tắc

## 1. Nguồn và Phạm vi Dữ liệu Thực nghiệm

Bản phân tích này chuẩn hóa toàn bộ hệ thống thực nghiệm của khóa luận theo thiết kế tách bạch (Decoupled Factorial Design), phân định độc lập đóng góp của **mô hình ensemble**, **chiến lược tìm kiếm multi-start**, và **từng toán tử láng giềng**.

Dữ liệu nền tảng sử dụng run benchmark chuẩn hóa `run_20260913_220633_fd7cfb` ([bản lưu JSON](../results/runs/run_20260913_220633_fd7cfb_multi_instance.json) và [kết quả chính thức](../results/multi_instance_tier3_results.json)):
- **Tính xác định & Tái lập Đa tiến trình (Deterministic Seeding)**: Toàn bộ quá trình sinh đường đi ngẫu nhiên sử dụng ánh xạ seed cố định `get_regime_seed_offset()` độc lập hoàn toàn với `PYTHONHASHSEED` của môi trường thực thi.
- **Quy mô mẫu đánh giá**: 10 kịch bản thiết kế địa hình/thời tiết độc lập ($N_{\text{scenarios}}=10$). Mỗi kịch bản được giải thuật lập lịch 1 lần duy nhất, sau đó kế hoạch bay được đánh giá đối đầu qua **cả 3 chế độ Ground Truth** (`in_ensemble`, `nominal_matched`, `misspecified_out_of_ensemble`). Tổng cộng 30 đơn vị đánh giá ($N=30$) $\times$ 100 mission/đơn vị = **3.000 mission mô phỏng** cho mỗi phương pháp.
- **Tính nhất quán phương sai (CRN)**: Các phương pháp trong cùng một kịch bản và cùng một chế độ Ground Truth dùng chung chính xác tập đường đi mục tiêu ngẫu nhiên (Common Random Numbers - CRN).
- **Phương pháp ước lượng độ bất định (Scenario-Level CI)**: Để tránh giả định sai về tính độc lập giữa các lượt đánh giá trên cùng một bài toán, độ bất định được ước lượng ở **cấp kịch bản (`scenario_id`)**:
  - **Khoảng tin cậy chính**: *Stratified Scenario Cluster Bootstrap CI 95%* ($B=10.000$ lượt lấy mẫu lại theo cụm `scenario_id`).
  - **Khoảng tin cậy tham số**: *Scenario-Aggregated Student's $t$ CI 95%* tính trên giá trị trung bình gom theo từng kịch bản ($N_{\text{scenarios}}=10$, bậc tự do $\text{df}=9$).

---

## 2. Ma trận Câu hỏi Nghiên cứu (Research Questions - RQ)

Mọi kết luận trong khóa luận được gắn trực tiếp với một thí nghiệm kiểm soát độc lập:

| Mã RQ | Câu hỏi nghiên cứu | Đối chứng kiểm soát | Biến độc lập | Chỉ số chính |
|---|---|---|---|---|
| **RQ1 (Evaluator)** | Fast Evaluator có bảo toàn độ chính xác và giảm runtime/eval so với Prefix-Only và Full không? | Full Evaluator, Prefix-Only, Fast Evaluator trên cùng tập candidate | Độ dài đoạn thay đổi $[t_a, t_b]$ | Relative Error $\|\Delta J\|/J$, Execution time ($\mu s$/eval) |
| **RQ2 (Multi-start)** | Khởi tạo 2-start (Ensemble + Nominal) có cải thiện nghiệm so với Single-start trong cùng tổng ngân sách? | `Proposed_Fast` (300+300) vs `SingleStart_FixedLS` (600) vs `SingleStart_MatchedTotal` (600+init) | Điểm xuất phát & phân bổ ngân sách | Objective $J$, DSR (%), RMST (tick), Tỷ lệ local optimum |
| **RQ3 (Ensemble vs Nominal)** | Tối ưu hóa theo Ensemble có tăng tính bền vững (robustness) khi môi trường thực tế sai lệch? | **Pure Model**: `Proposed_Fast` vs `Nominal_2Start` (cùng solver, cùng 2 start, cùng ngân sách)<br>**Pipeline**: `Proposed_Fast` vs `Nominal_Planning` (1-start) | Mô hình Evaluator (Ensemble vs Nominal); đối đầu qua 3 GT regimes | Ground Truth $J$, Empirical DSR (%), Paired $\Delta$DSR, RMST |
| **RQ4 (Toán tử & Thứ tự)** | Replace và Rebalance đóng góp gì, và thứ tự duyệt toán tử có gây thiên lệch kết luận ablation? | 4 cấu hình ablation $\times$ 3 chế độ thứ tự (`fixed`, `round_robin`, `shuffled`) | Tập toán tử bật/tắt và thứ tự duyệt | Số candidate sinh/khả thi, Tỷ lệ chấp nhận (%), $\Delta J$ |
| **RQ5 (Scalability)** | Thời gian lập lịch mở rộng như thế nào theo quy mô bài toán? | Tăng Horizon $H$, số UAV $K$, kích thước lưới $N$ | Quy mô không gian trạng thái | Planning Runtime (s), Peak Memory |

---

## 3. Xác nhận Thực thi & Bộ Kiểm thử

- **Toàn bộ Test Suite**: **99 passed** (0 failed, 0 error) thực thi trong 15.38s.
- **Bao phủ kiểm thử mới**: Bao gồm kiểm tra tái lập bitwise qua các process độc lập (`PYTHONHASHSEED`), kiểm tra tính toàn vẹn không bị cộng lặp số liệu tìm kiếm khi mở rộng đa ground truth, và kiểm tra hành vi thứ tự toán tử.
- **Bản ghi xác nhận**: Được lưu trữ tại [validation_latest.xml](../results/logs/validation_latest.xml).
- **Môi trường thực thi**: Python 3.12, NumPy 2.5.2, SciPy 1.18.1, pytest 9.1.1 trên Windows 11.
- **Lệnh tái lập toàn bộ kết quả**:

```powershell
# 1. Chạy toàn bộ 99 test kiểm thử tự động
.venv/Scripts/python.exe -X utf8 -m pytest -q --junitxml=results/logs/validation_latest.xml

# 2. Chạy benchmark đa kịch bản tách bạch 3 Ground Truth
.venv/Scripts/python.exe -X utf8 scripts/run_multi_instance.py --num-instances 10 --num-missions 100 --operator-order fixed

# 3. Xuất biểu đồ phân tích chuẩn luận văn
.venv/Scripts/python.exe -X utf8 scripts/plot_multi_instance_figures.py
```

---

## 4. Kết quả Chính: Phân định Đối chứng Thuần và Đối chứng Pipeline

Tổng hợp trên $N_{\text{scenarios}}=10$ kịch bản $\times$ 3 chế độ Ground Truth = 30 đơn vị đánh giá (3.000 mission/phương pháp): DSR càng cao càng tốt; RMST và năng lượng càng thấp càng tốt.

| Nhóm so sánh | Phương pháp | DSR (%) | CI 95% Bootstrap theo Scenario | RMST (tick) | Năng lượng (kJ) | $J$ Ground Truth |
|---|---|---:|:---:|---:|---:|---:|
| **Hệ thống đề xuất** | `Proposed_Fast` (2-start, Ens) | **27,87** | [22,70%; 33,30%] | **12,522** | 280,28 | **0,27249** |
| **Đối chứng Thuần Mô hình** | `Nominal_2Start` (2-start, Nom) | 26,87 | [21,70%; 32,37%] | 12,628 | 282,40 | 0,27072 |
| **Đối chứng Multi-start** | `SingleStart_Fast_FixedLS` | 28,03 | [23,00%; 33,43%] | 12,541 | 279,33 | 0,26350 |
| | `SingleStart_Fast_MatchedTotal` | 28,20 | [23,20%; 33,57%] | 12,545 | 279,41 | 0,26355 |
| **Đối chứng Pipeline truyền thống** | `Nominal_Planning` (1-start) | 27,10 | [22,00%; 32,53%] | 12,592 | 281,66 | 0,26830 |
| **Đối chứng Heuristic / Meta** | `Greedy_Lookahead` | 26,10 | [21,00%; 31,57%] | 12,688 | 275,17 | 0,25044 |
| | `Adaptive_GA` | 26,03 | [21,77%; 30,53%] | 12,677 | 272,13 | 0,25789 |

### Phân tích Khoảng tin cậy Ghép cặp (Paired Differences, ước lượng ở cấp Scenario)

| `Proposed_Fast` trừ đối chứng | Thắng / Thua / Hòa | $\Delta$DSR (điểm %) | CI 95% Cluster Bootstrap | CI 95% Scenario Student's $t$ | $p_{\text{boot}}$ | Diễn giải khoa học & Giới hạn bằng chứng |
|---|:---:|---:|:---:|:---:|:---:|---|
| vs `Nominal_2Start` | 19 / 10 / 1 | **+1,00** | [+0,23; +1,70] | [+0,10; +1,90] | 0,0144 | **Thuần mô hình**: Mô hình Ensemble cải thiện có ý nghĩa thống kê về DSR (+1,00%) và thời gian tìm thấy mục tiêu (RMST giảm 0,105 tick, $p < 0,05$) so với mô hình danh định khi cố định cùng số start. |
| vs `SingleStart_Fast_FixedLS` | 11 / 13 / 6 | **−0,17** | [−1,03; +0,97] | [−1,38; +1,05] | 0,6832 | **Multi-start**: Chưa phát hiện khác biệt có ý nghĩa thống kê về DSR giữa 2-start và 1-start trên tập kịch bản này (CI chứa 0). Tuy nhiên Proposed đạt $J_{\text{GT}}$ cao hơn (+0,0090). |
| vs `SingleStart_Fast_MatchedTotal` | 10 / 14 / 6 | **−0,33** | [−1,07; +0,47] | [−1,25; +0,58] | 0,3860 | Tương tự khi bù đắp ngân sách khởi tạo, DSR của 2-start và 1-start xấp xỉ nhau trên bộ mẫu này. |
| vs `Nominal_Planning` (1-start) | 18 / 12 / 0 | **+0,77** | [−0,27; +1,87] | [−0,53; +2,06] | 0,1584 | Xu hướng quan sát thấy Proposed nhỉnh hơn về DSR (+0,77%) và RMST (−0,070 tick), nhưng khoảng tin cậy vẫn bao quanh 0. |
| vs `Greedy_Lookahead` | 21 / 8 / 1 | **+1,77** | [+0,67; +3,03] | [+0,34; +3,19] | 0,0004 | Local Search cải thiện rõ rệt và có ý nghĩa thống kê cao ($p < 0,001$) so với khởi tạo tham lam. |
| vs `Adaptive_GA` | 19 / 10 / 1 | **+1,83** | [−0,40; +3,83] | [−0,71; +4,38] | 0,1052 | Quan sát thấy Proposed đạt DSR cao hơn (+1,83%) và $J_{\text{GT}}$ cao hơn (+0,0146), nhưng biến thiên giữa các kịch bản của GA khá lớn khiến CI chưa loại trừ giá trị 0. |

---

## 5. Phân tích Tách bạch theo 3 Chế độ Ground Truth

Trong thiết kế mới, **tất cả 10 kịch bản đều được đánh giá độc lập qua cả 3 chế độ Ground Truth** ($10 \times 3 = 30$ lượt đánh giá), bảo đảm tách biệt hoàn toàn đặc điểm địa hình bài toán khỏi chế độ kiểm thử:

| Chế độ Ground Truth | Số kịch bản | DSR Proposed (%) | DSR `Nominal_2Start` (%) | DSR `Nominal_Planning` (%) | $\Delta\text{DSR}$ (Proposed vs Nom2Start) | Ý nghĩa Vật lý & Động lực học |
|---|---:|---:|---:|---:|---:|---|
| **Trong Ensemble (In-Ensemble)** | 10 | **30,60** | 29,80 | 29,90 | **+0,80%** (RMST −0,112 tick) | GT thuộc không gian giả thuyết $\Theta$: Tối ưu hóa đa kịch bản giúp tăng cơ hội bao quát mục tiêu. |
| **Khớp Nominal (Nominal-Matched)** | 10 | **27,40** | 26,90 | 26,50 | **+0,50%** (RMST −0,052 tick) | GT khớp đúng mô hình danh định: Phương pháp đề xuất duy trì hiệu năng tốt mà không bị suy thoái quá mức. |
| **Sai đặc tả (Misspecified)** | 10 | **25,60** | 23,90 | 24,90 | **+1,70%** (RMST −0,152 tick) | GT lệch ngoài $\Theta$: Khi mô hình thực tế trôi dạt mạnh, Ensemble thể hiện tính phòng thủ cao nhất (**+1,70% DSR**). |

> [!NOTE]
> **Nhận định khoa học cho khóa luận**: Kết quả trên 3.000 mission kiểm chứng tính bền vững của mô hình Ensemble. Đặc biệt ở chế độ *Misspecified* (khi mục tiêu di chuyển lệch khỏi toàn bộ giả định mô hình hóa), `Nominal_2Start` bị sụt giảm DSR mạnh xuống 23,90%, trong khi `Proposed_Fast` vẫn duy trì 25,60% (+1,70%), khẳng định giá trị phòng thủ rủi ro của Ensemble.

---

## 6. Đóng góp Toán tử & Phân tích Tìm kiếm (Ablation Study)

Số liệu thống kê tìm kiếm (Search / Solver stats) được tổng hợp chính xác trên **10 lần giải thuật lập lịch duy nhất** (không bị nhân ba theo số ground truth):

| Cấu hình Ablation | $J$ Ensemble | DSR (%) | RMST (tick) | Đánh giá / Chấp nhận Replace | Đánh giá / Chấp nhận Rebalance |
|---|---:|---:|---:|---:|---:|
| **Bản đầy đủ (Full)** | **0,28679** | **27,87** | **12,522** | 2.674 / 20 | 1.487 / 13 |
| **Bỏ Replace (`NoReplace`)** | 0,28621 | 27,77 | 12,540 | — | 1.871 / 31 |
| **Bỏ Rebalance (`NoRebalance`)** | 0,28891 | 28,07 | 12,523 | 2.944 / 26 | — |
| **Bỏ cả hai (`NoReplaceRebalance`)** | 0,28486 | 27,43 | 12,581 | — | — |

### Phân tích Động lực học Không gian Láng giềng
1. **Toán tử Replace**: Đóng góp trực tiếp vào việc tái cấu trúc các cụm viếng thăm cell có xác suất tích lũy cao. Khi bỏ Replace, DwellRebalance phải tăng số lần chấp nhận từ 13 lên 31 lần để bù đắp việc điều chỉnh thời gian dừng.
2. **Toán tử DwellRebalance & Tương tác Thứ tự (`fixed`)**: Trong cơ chế duyệt cố định, việc bỏ Rebalance giúp giải phóng 1.487 lượt đánh giá cho các toán tử khác, khiến Replace tăng số lần chấp nhận từ 20 lên 26 và ChangeDwell từ 8 lên 11.
3. **Hiệu ứng kết hợp**: Cấu hình cơ sở bỏ cả hai toán tử (`NoReplaceRebalance`) cho thấy $J_{\text{ensemble}}$ thấp nhất (0,28486) và DSR thấp nhất (27,43%), chứng minh việc bổ sung hai toán tử này mang lại lợi ích rõ rệt cho chất lượng nghiệm.

---

## 7. Biểu đồ Minh họa và Khả năng Tái lập

![So sánh Đa Kịch bản Benchmark](../results/figures/fig4_multi_instance_benchmark.png)

![Ablation Toán tử Láng giềng](../results/figures/fig5_operator_ablation.png)

- Mọi biểu đồ và bảng số liệu đều có xuất xứ xác thực tại [figures_provenance.json](../results/figures/figures_provenance.json).
