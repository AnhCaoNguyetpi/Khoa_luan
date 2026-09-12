**Kế hoạch cải thiện proposal để phát triển thành bài báo**

Ngày rà soát: 08/09/2026. Đối tượng: `proposal.tex` và các module mô phỏng, belief, detection, planning, metrics trong repository hiện tại.

**Kết luận:** giữ bài toán wilderness SAR nhưng thu hẹp đóng góp chính về phân bổ thời gian tìm kiếm và định tuyến nhiều UAV dưới sai số của mô hình vị trí và khả năng phát hiện. Trước khi triển khai thêm thuật toán, cần sửa tính nhất quán của simulator, objective và baseline. “GIS + Bayes + nhiều UAV + replanning” chưa đủ xác lập novelty.

Đây là rà soát có mục tiêu, không phải systematic review bao phủ toàn bộ literature. Các kết luận về novelty dưới đây là định hướng cần kiểm chứng, không phải chứng nhận chưa có công trình tương tự. Phân biệt ba loại bằng chứng: nội dung nguồn đã đọc; lỗi đã tái hiện ở code; đề xuất/suy luận của người rà soát. Không sửa proposal hoặc thuật toán trong đợt này.

**1. Những công trình làm thay đổi cách định vị đề tài**

| Nguồn | Điều đã xác minh | Hệ quả đối với proposal |
|---|---|---|
| [Ge, Jiang & Coombes, 2024](https://arxiv.org/html/2411.10148v1) — preprint, đọc toàn văn HTML | Kết hợp hành vi tác tử, probability map, multi-UAV receding horizon và phân vùng động. Mục III-A giả định POD = 1; phần kết luận dành POD phụ thuộc địa hình cho tương lai. | Đây là đối chứng gần nhất. Khác biệt cụ thể có thể bắt đầu từ imperfect detection; không được suy ra rằng toàn bộ literature đều thiếu yếu tố này. |
| [Hashimoto et al., 2022](https://www.nature.com/articles/s41598-022-09502-4) — đọc toàn văn và kiểm tra supplementary | Hiệu chỉnh mô hình hành vi trên 65 sự cố hiker được chọn lọc. Có loại bỏ sự cố tìm thấy trong 1 km và gần biên. | Nguồn để hiệu chỉnh/kiểm tra hành vi, nhưng cần công bố selection bias và không mặc nhiên chuyển kết luận sang Việt Nam. |
| [Šerić et al., 2021](https://www.mdpi.com/2220-9964/10/2/80) — nội dung bài truy xuất được | Học tốc độ từ GPS của cứu hộ rồi hiệu chỉnh bằng dữ liệu người mất tích; mô tả 20 cặp vị trí đầu/cuối. | Dữ liệu người đi bộ bình thường và dữ liệu người mất tích cần được phân biệt; ít dữ liệu vẫn có thể dùng mô hình nhỏ có hiệu chỉnh. |
| [Ewers et al., 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1527095/full) — đọc toàn văn | Đã đánh giá DRL cho tìm kiếm wilderness dựa trên thông tin tiên nghiệm và các chỉ số tìm kiếm. | “Dùng ML để tăng hiệu quả tìm kiếm” quá rộng. Không cần thêm RL để chạy theo mức độ phức tạp. |
| [Dumenčić et al., 2025](https://www.mdpi.com/2504-446X/9/7/473) — nội dung nhà xuất bản truy xuất được, đối chiếu [preprint](https://arxiv.org/abs/2502.17372) | Kết hợp HEDAC, probabilistic detection và computer vision; có thử nghiệm thực địa với 78 tình nguyện viên. Bài dẫn dữ liệu OSF. | Imperfect detection gắn với điều khiển đã có nghiên cứu thực nghiệm; cần học cách kiểm chứng q và tìm baseline phù hợp. |
| [SAREnv, 2025](https://github.com/namurproject/SAREnv) — đọc repository chính thức; [bài báo](https://doi.org/10.3390/drones9090628) | Benchmark mở gồm 60 kịch bản địa lý; vị trí nạn nhân tổng hợp từ mô hình thống kê; có bốn baseline chính. | Có thể dùng benchmark ngoài hệ tự xây. Không gọi đây là 60 sự cố SAR thực hoặc quỹ đạo người mất tích quan sát được. |
| [Berger, Barkaoui & Lo, JORS 72(3), 2021; online 2020](https://www.tandfonline.com/doi/full/10.1080/01605682.2019.1685362) — abstract nhà xuất bản | Open-loop có anticipated feedback; tối thiểu hóa xác suất không phát hiện bằng formulation MILP và terminal belief. | Cần baseline tính trước có dự báo cập nhật âm tính. Không coi static đồng nghĩa giữ nguyên reward ban đầu. Chưa kiểm tra chi tiết formulation trả phí. |
| [Brown, 1980](https://pubsonline.informs.org/doi/10.1287/opre.28.6.1275) — abstract nhà xuất bản | Optimal search cho mục tiêu di chuyển trong không gian/thời gian rời rạc với search effort và detection hàm mũ. | Moving target, search effort và detection probability là nền tảng cũ; cần trích dẫn trước khi đặt tên đóng góp. |
| [Bertuccelli & How, CDC 2005](https://skoge.folk.ntnu.no/prost/proceedings/cdc-ecc05/pdffiles/papers/2737.pdf) — bản proceedings | Xét bản đồ xác suất không chính xác và quyết định tìm kiếm dưới bất định. | Robustness không phải novelty tự thân; cần xác định cấu trúc sai số và quyết định SAR cụ thể. |
| [Niu, Yi & Hong, 2026](https://www.mdpi.com/1424-8220/26/16/5189) — abstract/early-access nhà xuất bản | Bayesian search phân tán với detector phụ thuộc khoảng cách và mục tiêu tĩnh có tương quan không gian. | Cập nhật literature 2026; tránh tuyên bố tổ hợp Bayesian inference, detection và coordination chưa có. Chưa đủ toàn văn để kết luận về mọi ràng buộc của bài. |
| [Optimization of fleet search on network of regions, 2026](https://www.sciencedirect.com/science/article/pii/S0305054826000122) — abstract nhà xuất bản | Đồng thời phân bổ thời gian, thứ tự vùng và xác suất phát hiện có trọng số thời gian. Bối cảnh khác wilderness SAR. | Chỉ đối chiếu cấu trúc toán; không chuyển các giả định ứng dụng khác sang người mất tích. Joint routing–search duration cũng cần novelty cụ thể hơn. |
| [Elmachtoub & Grigas, Smart “Predict, then Optimize”](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922) — abstract nhà xuất bản, volume 2022 | Chất lượng quyết định được đưa vào quá trình học thay vì chỉ tối thiểu hóa sai số dự báo. | RQ “prediction tốt hơn có quyết định tốt hơn không?” có nền lý thuyết sẵn. Đánh giá downstream chưa đồng nghĩa thực hiện decision-focused learning. |

Khi cập nhật related work, lập ma trận chi tiết theo: mục tiêu tĩnh/động; p có học hay không; q hoàn hảo/ước lượng; dwell cố định/biến; UAV đồng nhất/khác nhau; nguồn thông tin online; pin/quay về; loại objective; dữ liệu và mã mở; kiểm thử ngoài phân phối. Ô chưa xác minh phải ghi “chưa xác minh”, không ghi “không có”.

**2. Điều chỉnh quan trọng đối với nhận xét trước: giá trị thực sự của replanning**

Suy luận từ chính mô hình đề xuất: giả sử chỉ có một mục tiêu; phát hiện chắc chắn thì dừng; trước khi dừng mọi quan sát đều là không phát hiện; chuyển động UAV và thời lượng hành động xác định; mô hình chuyển động mục tiêu và q đã cho; không có clue, quan sát môi trường hay thông tin ngoài mới.

Với một policy xác định, trước khi tìm thấy chỉ có một nhánh quan sát tiếp diễn: toàn bộ là âm tính. Ta có thể chạy policy đó trước nhiệm vụ trên nhánh này và ghi lại toàn bộ lịch hành động. Lịch tính trước, có quy tắc dừng khi tìm thấy, sẽ thực hiện đúng các hành động của policy cho tới lúc dừng. Mục tiêu di chuyển ngẫu nhiên nhưng không được quan sát thêm cũng không tự làm xuất hiện nhánh thông tin mới. Với policy ngẫu nhiên, có thể cố định seed trước nhiệm vụ theo lập luận tương tự.

Đây là lập luận về giá trị thích ứng trong các giả định trên, không phải khẳng định bài toán dễ giải. Open-loop tối ưu có thể rất khó tính; rolling horizon vẫn có giá trị về khả năng giải bài toán và đáp ứng thời gian. Nhưng thắng một baseline giữ nguyên p chưa chứng minh được value of information.

Sửa RQ4 theo hai mức:

1. **Bài báo chính:** đo trade-off chất lượng nghiệm–thời gian tính giữa anticipated-update open-loop và rolling horizon trong cùng mô hình, cùng nguồn lực.
2. **Mở rộng có điều kiện:** nếu muốn đo lợi ích thích ứng thông tin, thêm đúng một nguồn thông tin có ý nghĩa thực tế, ví dụ báo cáo vị trí có sai số đến giữa nhiệm vụ, hoặc quan sát chất lượng hình ảnh giúp cập nhật khả năng phát hiện. Khi đó đặc tả likelihood, thời điểm nhận và độ tin cậy; không cho planner biết trước realization.

Không thêm clue chỉ để tạo chiến thắng cho thuật toán. Nếu dữ liệu không hỗ trợ, giữ kết luận ở mức hiệu quả tối ưu hóa online. Bất định tham số có sẵn từ đầu cũng không tự phá được lập luận một nhánh nếu chưa có tín hiệu mới phân nhánh.

**3. Hướng bài báo nên chọn**

| Hướng | Đóng góp dự kiến | Điều kiện | Đánh giá |
|---|---|---|---|
| A. Phân bổ effort và routing có xét sai số p, q | Một formulation nhất quán, thuật toán phù hợp, phân tích lợi ích và failure regimes trên benchmark ngoài | Simulator đúng; baseline mạnh; sai số có cấu trúc; có hiệu chỉnh tối thiểu | Ưu tiên cho bài đầu tiên |
| B. Adaptive search nhờ thông tin mới | Định lượng value of information và chính sách replan | Có likelihood của clue/observation mới; open-loop đối chứng biết phân phối nhưng không biết realization | Chỉ làm nếu có căn cứ dữ liệu |
| C. Decision-focused belief learning | Học tham số để cải thiện quyết định tìm kiếm | Dữ liệu đủ; objective ổn định; kiểm thử ngoài vùng; chi phí huấn luyện chấp nhận được | Hướng tiếp theo, chưa nên đồng thời với A |

Tên làm việc cho A: **Detection-Aware Multi-UAV Search and Effort Allocation under Belief and Sensor Model Errors**. Chỉ thêm “Data-Driven” khi có phần học/hiệu chỉnh được mô tả và kiểm chứng. Nếu dữ liệu học chủ yếu là tổng hợp, ghi điều đó ngay trong abstract và experimental setup.

Đóng góp ứng viên: (i) tích hợp search effort với xác suất không phát hiện tích lũy và tính khả thi của tuyến; (ii) thuật toán có đối chiếu nghiệm tham chiếu trên bài nhỏ; (iii) kết quả ngoài vùng/bộ sinh chỉ ra khi nào sai số q làm mất lợi ích của detection-aware routing. Ba điểm này là mục tiêu phải đạt, không phải các kết quả hiện đã có.

**4. Problem statement đề xuất và các giả định cần chốt**

> Xét nhiệm vụ tìm kiếm một người mất tích trên đồ thị không gian của một khu vực wilderness bằng đội UAV có năng lượng hữu hạn. Vị trí mục tiêu là trạng thái ẩn với phân bố ban đầu và mô hình chuyển động được ước lượng từ dữ liệu hoặc mô hình hành vi đã hiệu chỉnh. Xác suất phát hiện phụ thuộc vào vị trí tương đối của UAV và mục tiêu, loại cảm biến, môi trường và thời lượng tìm kiếm. Hệ thống lựa chọn phân công, thứ tự ghé thăm và thời lượng tìm kiếm để tối đa hóa xác suất phát hiện trong thời hạn, đồng thời bảo đảm đường bay khả thi và đủ năng lượng quay về. Nghiên cứu đánh giá chất lượng quyết định khi các mô hình vị trí và phát hiện không chính xác, và tách lợi ích của mô hình, thuật toán giải và thông tin online.

Đề xuất mặc định cho bài đầu: một mục tiêu có thể di chuyển; stationary là trường hợp kiểm chứng; điều phối tập trung; liên lạc tức thời; số UAV nhỏ; hai cấu hình cảm biến nếu q có căn cứ, nếu không dùng một cấu hình với biến thiên theo môi trường; độ cao cố định; đường bay 2D theo hành lang khả thi có sẵn; không nhận obstacle avoidance 3D là đóng góp.

Chốt các lựa chọn bằng bảng assumptions trong proposal:

| Thành phần | Quyết định đề xuất |
|---|---|
| Mốc thời gian | t = 0 là lúc bắt đầu tìm; IPP thuộc t = -Δ; dùng Δ để sinh initial belief |
| Miền tìm kiếm | Đồ thị mục tiêu và đồ thị UAV riêng; vùng nước có thể cấm người đi bộ nhưng không tự động cấm UAV bay qua |
| Biên bản đồ | Mô phỏng miền đệm đủ rộng và theo dõi mass ngoài vùng tìm; nếu thêm trạng thái outside phải định nghĩa khả năng quay lại |
| Hành động | Bay, tìm, chờ, quay về; thời lượng tìm thuộc tập nhỏ đã định trước |
| Quan sát | Xác nhận phát hiện thì dừng; false positive bỏ qua trong mô hình đầu và ghi rõ. Không gọi raw detector confidence là xác nhận |
| Quan sát khi đang bay | Mặc định không tính; nếu có thì footprint và q phải áp dụng liên tục trên đường bay |
| Năng lượng | Ràng buộc cứng cộng reserve để quay về tại mọi thời điểm |
| Sortie | Kiểm chứng single-sortie trước; multi-sortie với một depot, thời gian thay pin cố định là thử nghiệm mở rộng chung cho mọi arm |
| Thông tin thời tiết | Planner dùng thông tin hiện tại và forecast được cho phép; không đọc toàn bộ thời tiết tương lai như đã biết |
| Thời điểm replan | Theo chu kỳ hoặc khi UAV rảnh, cố định trong thí nghiệm; chưa tối ưu trigger |
| Heterogeneity | Chỉ khác những thông số đã công bố; so sánh thuật toán trên cùng fleet trước khi so sánh cấu hình fleet |

**5. Formulation cần thay đổi**

Ký hiệu a_t là joint action, L_t là vị trí mục tiêu. Đặt b_t^-(i) là belief trước quan sát, b_t^+(i) là belief sau quan sát và M_t(i,j) là xác suất chuyển từ i sang j trong một tick. Một tick cần có thứ tự duy nhất: dự báo → hành động/quan sát theo quy ước → cập nhật; áp dụng thống nhất ở planner và simulator.

Thay q_ikt bằng q_{kit}(a_{kt}) nếu cần biểu diễn detection tại ô mục tiêu i khi UAV k thực hiện hành động ở vị trí khác. Footprint phải xuất hiện trong likelihood, không chỉ trong visualization.

Với các sensor độc lập có điều kiện theo trạng thái đủ đầy, xác suất tất cả UAV không phát hiện tại ô i là:

\[
\ell_t(i,a_t)=\prod_k[1-q_{kit}(a_{kt})].
\]

Sau quan sát toàn âm tính:

\[
b_t^+(i)=\frac{b_t^-(i)\ell_t(i,a_t)}{\sum_j b_t^-(j)\ell_t(j,a_t)},
\qquad
b_{t+1}^-(j)=\sum_i b_t^+(i)M_t(i,j).
\]

Nếu che khuất là biến ẩn chung, tích độc lập chỉ là xấp xỉ: cần joint likelihood hoặc đưa trạng thái che khuất vào mô hình. Không áp dụng tích này như một tính chất luôn đúng.

Để đánh giá một lịch hành động tính trước trên nhánh chưa phát hiện, dùng mass chưa chuẩn hóa u:

\[
u_0^-=b_0,\quad u_t^+(i)=u_t^-(i)\ell_t(i,a_t),\quad
u_{t+1}^-(j)=\sum_i u_t^+(i)M_t(i,j).
\]

Với H tick tìm kiếm t=0,...,H-1 và miền trạng thái bảo toàn xác suất:

\[
J_H(a)=P(T_D\le H\Delta t)=1-\sum_i u_{H-1}^+(i).
\]

Công thức này tránh cộng lặp p·q qua thời gian/UAV. Nếu có các nhánh thông tin online khác, phải lấy kỳ vọng trên các nhánh đó; không dùng nguyên một chuỗi toàn âm tính để tính toàn bộ giá trị policy.

Objective chính: max J_H với constraints nguồn lực. Dùng xác suất phát hiện sớm và thời gian giới hạn làm chỉ số phụ. Chưa cần cộng nhiều penalty có đơn vị khác nhau. Revisit không mặc nhiên là lãng phí: imperfect detection và target motion có thể khiến tìm lại hợp lý. Loại bỏ hoặc ablate penalty overlap khi belief đã phản ánh các lượt tìm trước.

Search effort: bắt đầu với d thuộc {1,2,4} tick, coi đây là cấu hình pilot cần hiệu chỉnh. Có thể dùng q(d)=1-exp(-λdΔt) dưới giả định hazard phù hợp. Nếu có nhóm mục tiêu bị che khuất kéo dài, đánh giá thêm mô hình q(d)=ρ[1-exp(-λdΔt)] hoặc trạng thái visibility ẩn; phải xác định sự phụ thuộc giữa các lượt tìm, không reset visibility tùy tiện để q tiến về 1.

Ràng buộc tối thiểu: tính liên tục của tuyến; thời gian tới trước khi tìm; mỗi UAV chỉ một hành động; dwell và travel time có đơn vị; đủ pin cho route cộng quay về; đường/cạnh khả thi; depot; horizon; quy tắc nhiệm vụ kết thúc; tài nguyên dùng sau khi phát hiện được tính riêng nếu cần quay về thực tế. Không dùng khoảng cách nền đất thay đổi cao độ làm độ cao bay nếu chưa có giả định terrain following.

**6. Những vấn đề ở code cần xử lý trước nghiên cứu thuật toán**

Đây là audit có mục tiêu, chưa phải review toàn bộ repository. Không suy ra rằng tất cả kết quả cũ đều sai; cần gắn từng kết quả với lỗi có thể tác động.

| Mức | Vị trí | Bằng chứng và tác động | Việc cần làm / tiêu chí nghiệm thu |
|---|---|---|---|
| P0 | `src/sar_uav/sim/metrics.py:64,75,95` | Đã chạy ví dụ 2 mission: một phát hiện phút 30, một không phát hiện, H=180. Kết quả DSR=0.5 nhưng P(T≤180)=1.0 và survival(H)=0. | CDF phải dùng indicator detected AND detect_time≤deadline. Giữ capped time cho RMST; failure không trở thành event tại H. Kiểm thử all-fail, all-success, mixed, event đúng deadline và horizon khác nhau. |
| P0 | `belief/motion.py::_move_scores` | Dùng feature ô nguồn `base[:,None]` cho mọi hướng, thay vì feature ô đích. Đã dựng 3 ô: bên trái trên trail, bên phải cách trail 1000 m; xác suất đi trái/phải đều 0.43641465. | Sửa score theo feature đích/edge để khớp phương trình và ý nghĩa attraction. Test hướng ưu tiên tới trail, giới hạn tốc độ, tính stochastic; không chỉ test hàng cộng bằng 1. |
| P0 | `sim/mission.py:268–287` | Observation sinh từng tick nhưng posterior áp q_dwell ở cuối lượt trong khi target đã di chuyển. Footprint bên cạnh dùng f·q_dwell thay vì 1-(1-fq_tick)^d. Với q=.3,f=.45,d=3 hai giá trị là .29565 và .352785375. | Cập nhật likelihood mỗi tick cùng tick sinh quan sát; nếu giữ observation theo block phải tích phân đường đi ẩn đúng. Test Monte Carlo khớp xác suất tính giải tích và thứ tự nhiều sensor. |
| P0 | `detection/sensors.py:17`, `experiments/datasets.py:1` | q ghi rõ là placeholder; dữ liệu học được sinh từ cùng quá trình tổng hợp của ground truth. | Tách calibrated/assumed parameters; test bộ sinh khác họ và vùng khác. Không gọi dataset tổng hợp là dữ liệu ISRID thực. |
| P1 | `planning/milp_pulp.py:61–78` | Vòng lặp giải từng UAV riêng; ordering nearest-neighbour. Chưa phải joint routing MILP. | Đổi tên trung thực thành allocation heuristic; muốn optimality gap phải có formulation tham chiếu chung objective, action space và constraints. Gap của surrogate không phải gap bài SAR gốc. |
| P1 | `sim/mission.py:224` | Seed observation phụ thuộc arm_name; common random numbers hiện có cho weather/target, chưa cho sensor noise. | Pairing theo scenario vẫn dùng được. Để giảm variance và ổn định khi đổi tên arm, dùng RNG sensor có chỉ số mission–time–sensor/measurement; kiểm thử marginals và conditional independence. |
| P1 | `planning/cycle_planners.py:92`, `milp_pulp.py:67` | Có penalty theo số lần ghé thêm vào belief đã cập nhật. | Ablate để đo xem có phạt lặp hai lần hoặc bỏ qua revisit có ích. Dùng residual detection mass làm cơ sở chính. |
| P1 | `data/areas/tay_nguyen_real/area.json` so với proposal | Dữ liệu đang có là Copernicus DEM, WorldCover, OSM, NASA POWER; proposal mô tả SRTM/ERA5-Land. | Viết đúng dữ liệu thực sự sử dụng; ghi thời gian, CRS, resampling, nguồn và hash. |

P0 là điều kiện trước khi tin kết quả; P1 là điều kiện trước khi diễn giải đóng góp. Chạy lại output bị ảnh hưởng sau sửa, giữ phiên bản kết quả cũ để truy nguyên. Không chỉnh mô hình theo hướng “làm dynamic thắng”.

**7. Kế hoạch dữ liệu thực tế và giới hạn của từng nguồn**

**Lost-person data.** Kiểm tra trực tiếp supplementary của Hashimoto: [CSV 1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-022-09502-4/MediaObjects/41598_2022_9502_MOESM1_ESM.csv) có 32.500 dòng dữ liệu, cột `incident_index,rep,closestpt_lat,closestpt_lon,closestpt_time_hr`; [CSV 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-022-09502-4/MediaObjects/41598_2022_9502_MOESM2_ESM.csv) có 65 dòng profile hành vi. CSV 3 chưa xác minh được nội dung vì phản hồi HTML. Không nhầm số replicate với số sự cố độc lập. Các closest points/profile đã fit không được dùng làm dữ liệu test độc lập hoặc làm profile “đã biết” của incident test.

Việc cần làm: lập data dictionary theo incident; xác minh IPP, find location, elapsed time có thực sự tồn tại; chia theo incident/nhóm địa lý trước fit. Nếu thiếu timestamp, không gán thời gian của closest point mô phỏng là thời gian tìm thấy quan sát được. Dữ liệu endpoint có thể kiểm chứng phân bố cuối/hit region, không đủ tự nhận đã xác nhận toàn bộ movement dynamics. Find location còn phụ thuộc quá trình tìm kiếm trước đó, nên cần thảo luận observation/selection bias.

**Detection data.** [WiSARD chính thức](https://sites.google.com/uw.edu/wisard/) cung cấp ảnh RGB/thermal, bản đầy đủ khoảng 40.54 GB và bản mẫu khoảng 971.6 MB. Đây là nguồn cho perception; chưa tự cung cấp mọi trial âm tính cần để ước lượng probability of detection trên mỗi lượt tìm. Dữ liệu thực nghiệm của Dumenčić được bài báo dẫn tại [OSF](https://osf.io/kb9e7/); chưa xác minh file do trang không truy xuất được trong phiên này.

Việc cần làm: kiểm tra sample/metadata trước tải toàn bộ; chia train/test theo chuyến bay, địa điểm, người và thời điểm để tránh frame gần nhau rò rỉ. Định nghĩa “opportunity to detect”, ground-truth presence, visibility, chiều cao, search duration và confirmation rule. Báo cáo calibration/Brier/log loss theo exposure cùng các nhóm điều kiện; mAP không thay thế q. Nếu ground truth chỉ bao gồm người nhìn thấy được, phải bổ sung mô hình xác suất nhìn thấy, hoặc giới hạn q là có điều kiện theo visibility.

**Benchmark ngoài.** Dùng SAREnv theo hai chế độ: (i) tái lập baseline và metric gốc để kiểm tra adapter; (ii) benchmark mở rộng có battery/detection/motion, công bố mọi thay đổi và chạy lại tất cả thuật toán trong cùng simulator. Không so trực tiếp số DSR của hai chế độ. Chia theo vùng địa lý, không chia các crop chồng lấn của cùng địa điểm sang train/test. Moving-target extension là phần bổ sung của nghiên cứu, không phải dữ liệu trajectory thực của benchmark.

**GIS Việt Nam.** Giữ Tay Nguyen hiện có làm case study ban đầu và bổ sung vài vùng có cấu trúc đường mòn/địa hình khác khi pipeline đã ổn. GIS thực không đồng nghĩa đã kiểm chứng khả năng phát hiện người thực. [WorldCover](https://worldcover2021.esa.int/data/docs/WorldCover_PUM_V2.0.pdf) là bản đồ lớp phủ, không phải phép đo trực tiếp mật độ tán che đối với camera UAV; hệ số chuyển land cover→visibility phải ghi là proxy/giả định và kiểm tra độ nhạy.

**Weather.** Theo [NASA POWER](https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/), độ phân giải khí tượng khoảng 0.5°×0.625°. Với vùng 4×4 km, không diễn giải nội suy thành thời tiết thực độc lập từng ô 100 m. Nếu dùng ERA5-Land, kiểm tra [catalog biến](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview); cloud cover cần nguồn có trường tương ứng, chẳng hạn [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview). Reanalysis tương lai chỉ dùng làm realized environment; forecast của planner cần tách riêng.

**8. Thuật toán đề xuất và baseline bắt buộc**

Không chọn ngay một metaheuristic rồi tìm objective phù hợp. Đầu tiên xây dựng evaluator J_H đúng và oracle cho bài nhỏ. Dùng exhaustive enumeration trên vài ô/UAV/tick hoặc formulation chính xác phù hợp; đây là test objective và upper bound/optimal solution, không phải planner production.

Cho bài lớn: sinh shortlist vùng dựa trên marginal detection gain, tạo route bằng insertion, cải thiện bằng relocate/swap/2-opt, lựa chọn dwell rời rạc và đánh giá lại residual mass theo thời điểm đến. Phân công UAV tiếp theo phải tính đóng góp tăng thêm sau các route đã được đặt; không cộng reward độc lập khi footprint chồng nhau. Dùng sparse transitions/particles khi lưới lớn; kiểm tra sai số xấp xỉ bằng evaluator tham chiếu.

| Baseline | Vai trò |
|---|---|
| B0 coverage / spiral | Mốc dễ hiểu, cùng ngân sách thời gian và pin |
| B1 greedy p | Tách ảnh hưởng của detection-aware score |
| B2 greedy marginal p·q / incremental time | Baseline thực dụng mạnh hơn chỉ chọn ô p cao |
| B3 anticipated-update open-loop | Tính trước motion, depletion và q; không nhận thông tin mới |
| B4 rolling nominal | Cùng evaluator, lookahead và candidate controls; đo hiệu quả replan |
| B5 một phương pháp literature gần | Ge-style RHC và/hoặc HEDAC tùy khả năng tái lập; mọi adaptation phải ghi rõ |
| B6 nghiệm tối ưu/upper bound bài nhỏ | Đo chất lượng nghiệm trên đúng formulation, không dùng allocation MILP hiện tại như oracle |

Nếu bổ sung planner có xét sai số, bắt đầu bằng ensemble hữu hạn θ_s=(p_s,M_s,q_s), đánh giá J_H(a;θ_s). So sánh nominal với mean-over-scenarios và, khi có lý do, lower-tail/worst-case. Chọn mức bảo thủ trên validation. “Worst-case trên tập kịch bản” không phải bảo đảm distributionally robust cho mọi phân phối. Nếu không xây planner robust, tên bài nên nói “under model errors”, không nói “robust optimization”.

Budget công bằng gồm thời gian tính offline, online và latency p95. Có hai so sánh khác nhau: cùng tổng compute và cùng deadline ra quyết định; công bố rõ so sánh nào. Một upper bound cho biết mô hình thật nhưng không biết trajectory tương lai khác với clairvoyant biết trajectory; không gộp hai oracle này.

**9. Viết lại RQ và chương trình thực nghiệm**

Giảm 5 RQ thành 3 câu trung tâm:

1. **RQ-A:** joint effort–routing có tăng J_H so với fixed dwell và detection-blind routing trên cùng fleet/nguồn lực không? Lợi ích phụ thuộc mức không đồng nhất q và chi phí tiếp cận thế nào?
2. **RQ-B:** sai số nào ở p, q, M làm thay đổi quyết định và gây mất hiệu quả lớn nhất? Planner xét nhiều mô hình có giảm mất mát ngoài phân phối không?
3. **RQ-C:** rolling horizon đổi chất lượng nghiệm lấy chi phí tính như thế nào so với anticipated-update open-loop? Nếu thêm thông tin online hợp lệ, đo riêng giá trị thông tin đó.

Prediction quality trở thành nhóm thí nghiệm hỗ trợ RQ-B. Không đặt giả thuyết bắt buộc mọi cải thiện log loss phải làm DSR tăng: thứ hạng quyết định có thể không đổi hoặc reward nằm ở nơi UAV không kịp đến.

| Thí nghiệm | Thiết kế | Kết luận được phép |
|---|---|---|
| E0 xác minh toán/simulator | Bài nhỏ analytic; q=0/1; sensor trùng; target tĩnh/động; no-detection replay; edge cases metrics | Pipeline tính đúng các đại lượng đã định nghĩa |
| E1 calibration/prediction | Holdout incident/flight/region; uniform, distance, GIS, fitted model | Khả năng dự báo trên phân phối/miền test đã nêu |
| E2 q-aware × dwell | 2×2: biết q/không dùng q trong planning × dwell cố định/tối ưu; filter giữ như nhau | Hiệu ứng chính và tương tác; không lẫn đổi phần cứng |
| E3 open-loop × rolling | Cùng belief/detection model và solver controls; có motion/depletion anticipation cho cả hai | Hiệu quả chiến lược giải; chỉ gọi value of information nếu có nhánh thông tin mới |
| E4 model errors | Sai số p, q, M riêng rồi tổ hợp một số mức đã chọn | Failure regimes và khả năng kháng sai lệch trong miền đã thử |
| E5 transfer | Vùng chưa thấy, bộ sinh khác họ, benchmark ngoài; Việt Nam là case study riêng | Khả năng khái quát trong các điều kiện đã kiểm chứng |
| E6 scalability | Tăng số ô/UAV/horizon; p50/p95 runtime và khoảng cách tới oracle khi có | Khả năng đáp ứng compute và trade-off chất lượng |

E4 không chỉ trộn p với uniform: thêm dịch đỉnh xác suất, thiếu một mode, prior quá tập trung, IPP lệch, sai trọng số nhóm hành vi, persistent occlusion, sai đặc tính sensor theo vùng. Thêm trường hợp hai model có log loss gần nhau nhưng lỗi nằm ở vùng có cost/reachability khác nhau. Không hiệu chỉnh simulator ground truth cho phù hợp với planner.

Đánh giá sai số q tách hai đường: q dùng trong planning và q dùng trong filtering. Thiết kế 2×2 đúng/sai ở hai đường giúp phân biệt chọn tuyến kém với loại nhầm probability mass. Khi đổi vegetation, nếu muốn cô lập detection thì giữ target trajectory; nếu vegetation ảnh hưởng cả chuyển động thì gọi đó là joint environment shift.

Số lần chạy: pilot khoảng 30–50 mission độc lập mỗi cấu hình được chọn; dùng variance của chênh lệch ghép cặp để tính cỡ mẫu chính, không xem 24 replicate là đủ mặc định. Ví dụ một tỷ lệ đơn ở trường hợp p≈0.5 cần khoảng 385 mẫu cho sai số xấp xỉ ±5 điểm phần trăm ở mức 95%; đây không phải power calculation cho mọi phép so sánh ghép cặp.

Dùng paired differences theo mission, Wilson CI cho tỷ lệ, paired/bootstrap theo cụm region/incident phù hợp, McNemar cho outcome nhị phân cùng scenario. Tránh coi nhiều seed trong một bản đồ là nhiều địa điểm độc lập. Định trước main contrasts và điều chỉnh multiple comparisons khi cần; không tăng số mẫu chỉ đến lúc có p<0.05.

Metrics chuẩn:

\[
\widehat F_D(h)=N^{-1}\sum_r 1\{D_r=1,T_r\le h\},\quad
\widehat S_D(h)=1-\widehat F_D(h),\quad
\widehat{RMST}_H=N^{-1}\sum_r\min(T_r,H).
\]

Với failure đặt T_r=∞ trong công thức, không biến thành event ở H. DSR bằng F_D(H) khi dùng cùng horizon. RMST ở đây là thời gian chờ phát hiện giới hạn, không phải survival của con người. Nếu horizon khác nhau, chọn common evaluation horizon hoặc xử lý censoring phù hợp. Report thêm total energy, energy after detection/return nếu mô hình yêu cầu, feasibility violations, revisit marginal gain, runtime, solver status và fallback count.

**10. Kế hoạch sửa từng phần proposal.tex**

| Phần hiện tại (dòng gốc) | Hành động cụ thể | Kết quả cần có |
|---|---|---|
| Động cơ (58) | Rút gọn bối cảnh; thêm trade-off xác suất vị trí–khả năng quan sát–chi phí tiếp cận | 2–3 đoạn dẫn vào gap cụ thể |
| Điểm xuất phát/giữ và thay đổi (104,136) | Gộp thành setting và notation; phân biệt t=-Δ và t=0 | Trạng thái, thông tin, tập hành động rõ |
| Bài toán trung tâm (189) | Viết lại problem statement; chốt assumptions | Input–decision–objective–constraints–output |
| Xác suất vị trí/phát hiện (219) | Bổ sung search action, footprint, dwell, conditional independence và calibration | q được định nghĩa đúng đơn vị exposure |
| Mô hình cơ sở (296) | Đổi objective sang mission detection; thêm unnormalized mass và routing feasibility | Formulation có thể kiểm chứng, không đếm trùng |
| Cập nhật/chuyển động (350,418) | Thống nhất thời gian; cập nhật từng tick; định nghĩa boundary và state đủ cho hành vi | Likelihood/motion khớp simulator |
| Dữ liệu (493–688) | Tách acquired/verified/planned; ghi rõ synthetic, selection bias, split | Data provenance và phương án dự phòng |
| Mô phỏng (692–861) | Tách truth/estimate; thêm RNG policy, no-detection replay, analytic validation | Simulator specification có thể tái lập |
| RQ (868) | Gộp RQ-A/B/C; thay lời khẳng định lợi ích thành giả thuyết kiểm chứng | Mỗi RQ ánh xạ đúng experiment |
| Thực nghiệm (1014) | Thêm mạnh baseline B3/B5/B6, factorial và out-of-region | Đo đúng hiệu ứng, tránh baseline yếu |
| Metrics (1158) | DSR/CDF/RMST, return cost, runtime và CI | Không đếm failure thành detected |
| Đóng góp (1200) | Chỉ nêu 2–3 điểm cụ thể, đánh dấu dự kiến | Khác biệt có nguồn và bằng chứng dự kiến |
| Phạm vi/kết quả (1245–1343) | Chốt must-have và conditional extensions; cho phép null results có thông tin | Tiêu chí hoàn thành không dựa vào “phải thắng” |
| References (1348) | Bổ sung optimal search, SAREnv, anticipated feedback, robust search, literature 2026 | Phân biệt preprint, online date và volume year |

Cấu trúc bài báo sau này: Introduction → Related work → Problem formulation → Method → Data and experimental protocol → Results → Limitations → Conclusion. Proposal có thể thêm work plan và feasibility; không cần mang toàn bộ các đoạn giải thích nhập môn sang paper.

**11. Lộ trình triển khai dự kiến 10 tuần**

Ước lượng cho một người đã có framework, không bao gồm thời gian xin dữ liệu hoặc tổ chức bay thực địa. Nếu nguồn lực khác, giữ thứ tự phụ thuộc và điều chỉnh thời lượng.

| Tuần | Việc chính | Deliverable và điều kiện chuyển bước |
|---|---|---|
| 1 | Chốt hướng A, assumptions, gap matrix, data audit; viết lại problem statement | Một formulation thống nhất; danh sách dữ liệu đã xác minh và giới hạn claims |
| 2 | Sửa metrics, motion, per-tick likelihood, footprint; audit clock/units/RNG | E0 qua các bài analytic; lưu provenance phiên bản và đánh dấu output cũ bị ảnh hưởng |
| 3 | Adapter benchmark, data split; pilot calibration q và initial belief | Tải/đọc được mẫu dữ liệu; split manifest cố định; quyết định giữ/bỏ Data-Driven trong tên |
| 4 | Objective evaluator, oracle bài nhỏ, anticipated-update open-loop | So sánh analytic/Monte Carlo; no-detection replay khớp trước khi tìm thấy |
| 5 | Joint effort–routing heuristic và nominal rolling | Feasible routes; kết quả pilot đối chiếu greedy và oracle; solver fallback được log |
| 6 | Baseline literature/adaptations; có thể bổ sung ensemble nếu bằng chứng cần | Bảng baseline công bằng; xác định main contrasts và compute budget |
| 7 | Pilot thống kê và chốt protocol trước test chính | Cỡ mẫu dựa trên effect/precision mục tiêu; freeze hyperparameters và test regions |
| 8 | E2–E5 chính; thu thập failure cases và null results | Mission-level results kèm CI, model versions, seeds, config hashes |
| 9 | Scalability, transfer, sensitivity grid/time/occlusion, xử lý failure | Runtime p95; stability qua độ phân giải; giải thích rõ khi phương pháp mất lợi thế |
| 10 | Viết paper draft, sửa proposal và reproducibility package | Claims khớp bảng/figure; một lệnh tái tạo kết quả chính; limitations trung thực |

Ba checkpoint tránh lãng phí: sau tuần 2 chưa đúng simulator thì chưa chạy grid lớn; sau tuần 3 chưa có dữ liệu phù hợp thì giảm claims data-driven; sau tuần 6 chưa hơn baseline mạnh thì phân tích nguyên nhân/failure regime trước khi thêm thuật toán phức tạp.

**12. Bộ kết quả tối thiểu để viết paper có sức thuyết phục**

1. Bảng related-work matrix có nguồn cho từng nhận định thiếu/có.
2. Bảng assumptions, nguồn dữ liệu và split protocol.
3. Figure giải thích belief, detectability, route và residual mass trên một mission.
4. Bảng analytic/oracle validation và optimality gap đúng formulation.
5. Main outcome table: detection trong H, RMST, energy, feasibility, runtime, CI.
6. Ablation q-aware × dwell, anticipated-update open-loop × rolling.
7. Robustness heatmap cho sai số p/q/M và biểu đồ transfer qua vùng.
8. Hai hoặc ba failure cases phân tích cơ chế, kể cả khi coverage/greedy tốt hơn.
9. Public configs/code hoặc gói tái lập phù hợp quyền dữ liệu; không cần công bố vị trí sự cố chi tiết nếu điều kiện dữ liệu không cho phép.

Tiêu chí paper-ready không phải “DSR tăng X%” được chọn sau khi xem dữ liệu. Cần có một đóng góp xác định, mô hình đúng, baseline công bằng, bằng chứng ngoài bộ sinh tự xây, mức cải thiện hoặc giới hạn có ý nghĩa thực tế, và kết quả tái lập. Nếu chỉ có simulation trên GIS thực với q giả định, bài vẫn có thể được trình bày như nghiên cứu phương pháp/mô phỏng, nhưng chưa thể tuyên bố hiệu quả triển khai cứu hộ thực địa.
