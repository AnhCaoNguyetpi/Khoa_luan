# Kiến trúc Hệ thống uav-sar

## Luồng dữ liệu và Tối ưu hóa tổng thể

Hệ thống được thiết kế theo đúng mô hình toán học và giải thuật đề xuất trong đề cương nghiên cứu:
> **Multi-UAV Routing and Search-Effort Allocation under Persistent Detection-Model Uncertainty**

```mermaid
flowchart TD
    subgraph INPUTS["1. Không gian, Mục tiêu và Giả thuyết Cảm biến"]
        GIS[Dữ liệu GIS Tây Nguyên / SAREnv<br/>Lưới G, đồ thị độ cao, ma trận khoảng cách]
        B0[Initial Belief b_0 và Ma trận chuyển động Markov M]
        HYP[Tập giả thuyết sai số cảm biến S = {q^s, w_s}<br/>log lambda_i^s = log hat_lambda_i + eta_{g(i)}^s]
    end

    subgraph ENGINE["2. Động cơ Joint Belief & Khối lượng Xác suất"]
        MASS[JointMassEngine]
        FWD[Tiền tố Forward Unnormalized Mass<br/>u_t^-, u_t^+ dọc theo lịch a]
        BWD[Hậu tố Backward Continuation<br/>V_t(i,s) với biên V_H = 1.0]
        OBJ[Hàm mục tiêu tích lũy ensemble:<br/>J_S(a) = 1 - sum u_{H-1}^+]
    end

    subgraph SEARCH["3. Thuật toán Joint Route--Effort Local Search (Algorithm 1)"]
        LS[JointRouteEffortLocalSearch]
        NB[NeighborhoodExplorer: 8 toán tử biến đổi<br/>Swap/2-opt, Insert revisit, Delete,<br/>Replace Visit, Relocate, Change Dwell,<br/>Dwell Rebalance, Adjust Wait]
        CHK[ConstraintChecker:<br/>Đẳng thức nối tiếp s_{kl}, ngân sách pin B_k - R_k,<br/>Kiểm tra trước chặng bay dở dang u->v->o]
        EVAL{3-Tier Evaluator}
        FAST[ForwardBackwardFastEvaluator<br/>Ghép u_{t_a}^- với V_{t_b+1}]
        PRE[PrefixOnlyEvaluator<br/>Tái sử dụng u_{t_a}^-, tính tiếp tới H-1]
        FULL[FullEvaluator<br/>Tính lại từ đầu 0->H-1]
    end

    subgraph SIM["4. Mô phỏng Thực thi & Protocol An toàn"]
        SIMU[ScheduleSimulator]
        CRN[IndexedRNGStream 4 phần tử:<br/>(mission, tick, uav, event)]
        ABORT[Quy tắc quay về an toàn khi phát hiện:<br/>UAV tại trạm về depot; UAV đang bay hoàn thành leg v rồi về]
        MET[Chỉ số thống kê chuẩn:<br/>DSR, RMST, Actual Energy, Calibration Gap]
    end

    GIS --> CHK
    GIS --> HYP
    B0 --> MASS
    HYP --> MASS
    MASS --> FWD & BWD --> OBJ
    OBJ --> EVAL
    EVAL --> FAST & PRE & FULL
    LS <--> NB
    NB --> CHK
    CHK --> LS
    LS --> EVAL
    LS -->|Lịch tối ưu a*| SIMU
    CRN --> SIMU
    SIMU --> ABORT --> MET
```

---

## Các Module Cốt lõi (`src/sar_uav/`)

### 1. `detection/hypothesis.py`
- Quản lý tập giả thuyết $\mathcal{S} = \{q^s, w_s\}_{s=1}^S$.
- Tạo ensemble sai số có cấu trúc không gian theo nhóm thảm thực vật $\eta_{g(i)}^s$.
- Tính hàm likelihood non-detection $\ell_t^s(i, a_t)$ vector hóa trên toàn bộ lưới ô.

### 2. `belief/joint_mass.py`
- Lớp `JointMassEngine`: Tính toán forward unnormalized mass $u_t^-(i, s), u_t^+(i, s)$, backward continuation $V_t(i, s)$ và fast evaluation cục bộ trong đoạn $[t_a, t_b]$.
- Đảm bảo tính tương đương đại số tuyệt đối giữa công thức nối mass và tính toán lại toàn bộ.

### 3. `planning/constraints.py`
- Lớp `ConstraintChecker`: Hiện thực hóa hệ thống đẳng thức thời gian (\ref{eq:cons1})--(\ref{eq:cons2}), ràng buộc deadline $H$ và năng lượng pin $B_k - R_k$.
- Thực hiện kiểm tra khả thi tường minh trước mỗi chặng bay dở dang: $B_k(t) \ge \bar e_{uv}^k + \bar e_{vo}^k + R_k$.

### 4. `planning/neighborhood.py`
- Lớp `NeighborhoodExplorer`: Sinh ứng viên tuần tự (lazy generation) qua 8 toán tử biến đổi (bao gồm `ReplaceVisit` và `DwellRebalance`).
- Hàm `detect_affected_interval`: Tự động nhận diện khoảng bị ảnh hưởng $[t_a, t_b]$ (nếu dịch lịch sau thì đặt $t_b = H - 1$).

### 5. `planning/evaluator.py`
- Cài đặt 3 cấp độ Evaluator: `FullEvaluator`, `PrefixOnlyEvaluator` và `ForwardBackwardFastEvaluator`.
- Thu thập thống kê chi tiết về thời gian đánh giá (ms), mức tăng tốc và phân bố độ dài đoạn ảnh hưởng $L_{\mathrm{affected}}$.

### 6. `planning/local_search.py`
- Cài đặt Algorithm 1 (`Joint Route--Effort Local Search`): Duyệt láng giềng theo trần số lượt đánh giá $N_{\max}$ và giới hạn thời gian $T_{\mathrm{wall}}$, cập nhật $J^*$ và làm mới cache.

### 7. `planning/baselines.py`
- Cài đặt các đối thủ thuật toán: `GreedyLookaheadPlanner`, `AdaptiveGAPlanner` và `ExactBruteForcePlanner`.

### 8. `sim/mission.py`
- Lớp `ScheduleSimulator`: Thực thi lịch $a^*$ trên mô hình mục tiêu ẩn dưới mô hình thực $q^\star$, kích hoạt quy tắc an toàn khi phát hiện và tính toán các metrics (DSR, RMST, Energy, Calibration Gap).
