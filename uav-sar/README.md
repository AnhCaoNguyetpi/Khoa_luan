# uav-sar — Data-Driven Belief-Adaptive Multi-UAV Search & Routing

Framework mô phỏng + tối ưu hóa cho đề tài:

> **Tìm kiếm và định tuyến động nhiều UAV dựa trên dữ liệu cho bài toán tìm
> kiếm khách du lịch mất tích trong môi trường tự nhiên**
> *(Data-Driven Belief-Adaptive Multi-UAV Search and Routing for Missing
> Tourists in Wilderness Areas)*

Toàn bộ framework cài đặt đầy đủ chu trình trong đề cương (`docs/proposal/proposal.tex`):

```
P^t → Phân công → Định tuyến → Tìm kiếm → Quan sát → Cập nhật Bayes → P^{t+1} → Tái tối ưu
```

Mission mặc định dùng lưới GIS thật `tay_nguyen_real` quanh 108.25°E,
12.60°N: Copernicus DEM GLO-30, ESA WorldCover 2021 v200,
OpenStreetMap và NASA POWER hourly. Provenance và SHA-256 được lưu trong
`data/areas/tay_nguyen_real/area.json`. `demo_valley` được giữ lại làm
fixture tổng hợp cho thí nghiệm cũ và test.

---

## 1. Cấu trúc thư mục

```
uav-sar/
├── configs/                  # cấu hình JSON (mission mặc định + từng RQ)
├── data/
│   ├── areas/demo_valley/    # grid + các lớp GIS (tự sinh / import)
│   └── models/               # trọng số PMR đã huấn luyện (*.npz)
├── docs/
│   ├── proposal/proposal.tex # đề cương nghiên cứu (bản gốc)
│   ├── architecture.md       # kiến trúc module & luồng dữ liệu
│   └── experiments.md        # ánh xạ RQ1..RQ5 ↔ code ↔ output
├── scripts/                  # entry-point chạy trực tiếp
├── src/sar_uav/
│   ├── config.py             # cấu hình mặc định + deep-merge loader
│   ├── data/                 # AreaData, terrain tổng hợp, weather, GIS adapters
│   ├── belief/               # 4 initial-belief models, Bayes update, motion model, PMR trainer
│   ├── detection/            # q_ikt theo sensor × môi trường × thời gian dwell
│   ├── uav/                  # UAVSpec (năng lượng), UAVState (state machine)
│   ├── planning/             # greedy, static multi-sortie, rolling-horizon, MILP (tuỳ chọn)
│   ├── sim/                  # target ẩn, MissionRunner, metrics DSR/EDT
│   ├── experiments/          # dataset incidents, belief eval, runner CRN, rq1..rq5, stats
│   └── viz/                  # matplotlib (Agg): heatmap belief, biểu đồ
├── tests/                    # 44 unit/integration tests + run_tests.py
└── results/                  # CSV + figure output (gitignored)
```

## 2. Cài đặt

```bash
cd uav-sar
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt                    # numpy, pandas, matplotlib
```

Tuỳ chọn: `pip install pulp` (planner MILP), `pytest` (chạy test kiểu pytest),
`rasterio geopandas shapely` (import dữ liệu thật).

## 3. Quickstart

```bash
# 0) dựng lại lưới GIS thật từ các file nguồn đã tải
python scripts/build_real_area.py

# 1) một mission demo (có snapshot PNG vào results/figures/demo)
python scripts/run_mission.py --planner rolling --belief gis

# 2) chạy thí nghiệm RQ4 (factorial static×dynamic) với 24 replicate
python scripts/run_experiments.py --rq 4 --replicates 24

# 3) toàn bộ RQ1..RQ5 (chạy lâu; thêm --quick để smoke-test)
python scripts/run_experiments.py --rq all
python scripts/summarize_results.py
```

Kết quả nằm trong `results/data/*.csv` (bảng mission-level + summary) và
`results/figures/*.png`.

## 4. Ánh xạ đề cương ↔ code

| Thành phần trong đề cương | Module |
|---|---|
| Grid `G`, spatial database ô `i` | `data/area.py` (`AreaData`) |
| Thời tiết ERA5-Land stand-in | `data/weather.py` |
| Initial belief: Uniform / Distance / GIS / **Data-driven** | `belief/models.py` |
| PMR logistic huấn luyện trên "ISRID tổng hợp" | `belief/train.py`, `experiments/datasets.py` |
| Motion model `M_ij` softmax + profile hành vi | `belief/motion.py` |
| Bayes update sau quan sát âm | `belief/update.py` |
| Detection `q_ikt` = f(sensor, veg, cloud, rain, wind, dwell) | `detection/sensors.py` |
| Năng lượng `γ·d (+η·climb)`, `α·τ`, reserve RTB, swap pin | `uav/platform.py`, `uav/state.py` |
| Objective `max Σ p·q·z − λ₁ travel − λ₂ energy − λ₃ overlap` | `planning/*` |
| Greedy / Static (open-loop, multi-cycle) / Rolling-horizon / MILP | `planning/greedy.py`, `cycle_planners.py`, `milp_pulp.py` |
| Ground truth ẩn ≠ belief của planner | `sim/target.py`, `sim/mission.py` |
| Quy trình mô phỏng 11 bước | `sim/mission.py::run_mission` |
| Metrics DSR, P(T≤T_max), EDT + chi phí vận hành | `sim/metrics.py` |
| RQ1–RQ5 (thiết kế + chỉ số đúng như đề cương) | `experiments/rq1..rq5.py` |
| Thống kê: Wilson CI, McNemar exact, paired bootstrap | `experiments/stats.py` |

## 5. Dữ liệu thật (tuỳ chọn)

Tải thủ công về máy rồi convert bằng một lệnh (không cần API key trong code):

| Lớp | Nguồn tải |
|---|---|
| DEM 30 m | SRTM 1-arc-second (Earthdata / OpenTopography) |
| Trails, roads | OpenStreetMap export (Geofabrik / overpass) → GeoJSON |
| Land cover 10 m | ESA WorldCover 2021 v200 (Zenodo) |
| Weather | ERA5-Land hourly (CDS) → CSV |

```bash
pip install rasterio geopandas shapely
python scripts/import_gis_layers.py --name taynguyen --cell 100 \
    --dem dem.tif --trails trails.geojson --roads roads.geojson \
    --landcover worldcover.tif --origin-lonlat 108.25 12.60
```

Sau đó đặt `"area": {"name": "taynguyen"}` trong config là mọi pipeline dùng
khu vực thật.

## 6. Ghi chú thiết kế (giả định đã tài liệu hoá)

* Planner **không bao giờ** thấy vị trí thật; mọi quyết định chỉ dùng `P^t`
  và mô hình ước lượng.
* Planner biết profile hành vi của mục tiêu (được perturb trong RQ5);
  việc "không biết profile" là hướng mở rộng.
* Footprint cảm biến > 1 ô: ô kề nhận `q × footprint_side` — phản ánh FOV
  thực tế so với độ phân giải grid.
* Quan sát âm dùng `q_dwell` tích luỹ trong thời gian dwell (xấp xỉ chuẩn
  trong tài liệu SAR probabilistic).
* RL/MARL cố tình **không** đưa vào — đúng phạm vi đề cương; planner ladder
  hiện có: greedy → static → rolling-horizon (→ MILP allocation tuỳ chọn).

## 7. Chạy test

```bash
python tests/run_tests.py      # không cần pytest
# hoặc: pytest tests/
```

44 test bao phủ: công thức Bayes đúng như đề cương, ma trận chuyển vị stochastic,
địa chất detection (monotonic, RGB vs thermal), năng lượng/state machine
(forced RTB + swap pin), tính khả thi của planner, mission end-to-end, metrics;
9 test thống kê xác minh Wilson CI khớp giá trị chuẩn, McNemar chính xác và
paired bootstrap phát hiện đúng hiệu ứng/giữ null.
