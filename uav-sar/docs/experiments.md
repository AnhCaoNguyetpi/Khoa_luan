# Ánh xạ RQ1–RQ5 ↔ code ↔ output

Mọi thí nghiệm chạy qua một entry-point duy nhất:

```bash
python scripts/run_experiments.py --rq all --replicates 24   # hoặc --quick để smoke-test
python scripts/summarize_results.py
```

Toàn bộ CSV nằm ở `results/`, log ở `results/logs/`. Mỗi arm trong mỗi RQ
chạy trên **cùng bộ môi trường theo replicate** (common random numbers,
xem `experiments/runner.py::make_setups`), nên mọi so sánh giữa các arm là
so sánh **có ghép cặp** (paired).

## Thống kê đi kèm (`src/sar_uav/experiments/stats.py`)

| Công cụ | Ý nghĩa |
|---|---|
| `wilson_interval(k, n)` | Khoảng tin Wilson 95% cho tỷ lệ (DSR, P(T≤deadline), top-k recall) |
| `mcnemar_exact_p(b, c)` | Kiểm định McNemar chính xác hai phía trên các cặp phân biệt (detected A vs B) |
| `compare_arms(rows, pairs)` | So sánh cặp theo replicate: diff = treatment − control, bootstrap CI 95% + p_boot; metric nhị phân có thêm p_McNemar |
| `arm_proportion_table` / `grouped_proportion_ci` | Bảng k/n + CI cho từng arm / từng điểm sweep |

Quy ước: pair = (control, treatment), diff dương = treatment cao hơn trên
metric đó (với `EDT_censored` thấp hơn mới tốt — đọc dấu diff tương ứng).

## RQ1 — Giá trị của dự báo vị trí dựa trên dữ liệu

* Câu hỏi: Uniform → Distance → GIS → Data-driven, bản đồ xác suất ban đầu
  nào tốt và giá trị chuyển thành vận hành thế nào?
* Code: `experiments/rq1.py`; huấn luyện PMR logistic bằng `belief/train.py`
  trên incidents tổng hợp (`experiments/datasets.py`).
* Config: `configs/experiments/rq1_probability_models.json`.
* Output:
  * `rq1_prediction_summary.csv` — loglik, rank, mass r∈{200,400,800} m,
    top-k recall (+ `_ci_lo/_hi` Wilson) trên tập test giữ riêng.
  * `rq1_prediction_rows.csv`, `rq1_initial_maps_example.png`,
    `rq1_mass_vs_radius.png`, `rq1_topk_recall.png`.
  * `rq1_operational_rows.csv` + `rq1_operational_summary.csv` — preview
    vận hành rolling-planner theo từng initial-belief model.
  * `rq1_operational_stats.csv`, `rq1_operational_comparisons.csv` —
    CI từng arm + so sánh cặp vs uniform.

## RQ2 — Chất lượng dự báo chuyển thành lợi ích vận hành?

* Câu hỏi: pha loãng bản đồ tham chiếu về uniform `P=(1−ε)P_ref+ε·noise`,
  DSR/EDT suy giảm ra sao?
* Code: `experiments/rq2.py`. Config: `rq2_prediction_to_routing.json`.
* Output: `rq2_rows.csv`, `rq2_summary.csv` (kèm `DSR_ci_lo/_hi`),
  `rq2_performance_vs_eps.png`,
  `rq2_comparisons.csv` (paired stats vs ε=0).

## RQ3 — Định tuyến có xét khả năng phát hiện?

* Câu hỏi: tiêu chí gán `p·q` (detection-aware) vs `p` (detection-blind),
  fleet hỗn hợp RGB+thermal vs thuần RGB.
* Code: `experiments/rq3.py` (fleet thuần RGB khai báo tại
  `HOMOGENEOUS_RGB_FLEET`). Config: `rq3_detection_aware.json`.
* Output: `rq3_rows.csv`, `rq3_summary.csv` (kèm CI),
  `rq3_dsr_by_veg.png`, `rq3_energy_by_veg.png`,
  `rq3_stats.csv` (pq vs p, paired trên cùng môi trường).

## RQ4 — Giá trị của tái định tuyến động (thiết kế 2×2)

```
                 Static   Dynamic
Baseline belief     A        B
Data-driven         C        D      (+ E: mục tiêu đứng yên)
```

* Contrasts: C−A (giá trị data-driven khi tĩnh), B−A (giá trị tái định
  tuyến với belief cơ sở), D−A (lợi ích framework đầy đủ), D−B (thêm
  data-driven khi đã động).
* E chỉ dùng cho ablation, **không** đưa vào kiểm định paired vì quy trình
  mục tiêu khác (đứng yên).
* Code: `experiments/rq4.py`. Config: `rq4_static_dynamic_factorial.json`.
* Output: `rq4_rows.csv`, `rq4_summary.csv`, `rq4_factorial_dsr.png`,
  `rq4_survival.png`, `rq4_stats.csv` + `rq4_arm_stats.csv`.

## RQ5 — Độ bền trước sai số mô hình

* Câu hỏi: planner cố tình dùng sai mô hình — `map_eps` (bản đồ sai),
  `q_bias` (detection bị over/under-estimate), `beta_bias` (motion model
  lệch) — trong khi môi trường và quá trình mục tiêu thật không đổi.
* Code: `experiments/rq5.py`. Config: `rq5_model_error.json`.
* Output: `rq5_rows.csv`, `rq5_summary.csv` (kèm `DSR_ci_lo/_hi`),
  `rq5_robustness_dsr.png`.

## Ghi chú diễn giải

* n=24 replicate/cell cho chênh lệch DSR nhỏ (~0.1) vẫn có CI rộng — hãy
  trích dẫn kèm CI và p-value trong bảng, tăng `--replicates` nếu cần bền
  hơn nữa (chi phí tuyến tính theo thời gian chạy).
* p_boot có sàn resolution 2/n_boot (5000 lần lấy mẫu mặc định); khi bảng
  ghi 0.0004 nghĩa là "< 2/n_boot".
* Các so sánh giữa arm cùng seed-set hợp lệ nhờ CRN; tuyệt đối không paired
  giữa các setup khác quy trình mục tiêu (ví dụ E vs A–D).
