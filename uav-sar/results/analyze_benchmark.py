import json
import statistics

file_path = r'c:\Users\ADMIN\Downloads\Khóa luận\uav-sar\results\multi_instance_tier3_results.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

instances = data.get('instances', [])
print(f"Total instances: {len(instances)}")

total_missions = 0
algos = ['SingleStart_Fast_FixedLS', 'Nominal_Planning', 'Greedy_Lookahead', 'Adaptive_GA']

wins = {alg: 0 for alg in algos if alg != 'SingleStart_Fast_FixedLS'}
losses = {alg: 0 for alg in algos if alg != 'SingleStart_Fast_FixedLS'}
ties = {alg: 0 for alg in algos if alg != 'SingleStart_Fast_FixedLS'}

metrics = {alg: {'detected': 0, 'energy': [], 'rmst': [], 'detection_tick': []} for alg in algos}

for inst in instances:
    missions = inst.get('mission_records', [])
    total_missions += len(missions)
    
    for mission in missions:
        results = mission.get('results', {})
        
        fixed_res = results.get('SingleStart_Fast_FixedLS', {})
        fixed_detected = fixed_res.get('detected', 0)
        
        for alg in algos:
            res = results.get(alg, {})
            metrics[alg]['detected'] += res.get('detected', 0)
            metrics[alg]['energy'].append(res.get('energy', 0))
            metrics[alg]['rmst'].append(res.get('rmst', 15.0))
            
            tick = res.get('detection_tick')
            if tick is not None:
                metrics[alg]['detection_tick'].append(tick)
                
            if alg != 'SingleStart_Fast_FixedLS':
                alg_detected = res.get('detected', 0)
                if fixed_detected > alg_detected:
                    wins[alg] += 1
                elif fixed_detected < alg_detected:
                    losses[alg] += 1
                else:
                    ties[alg] += 1

print(f"Total missions evaluated: {total_missions}")
print("\n--- Win/Loss/Tie vs Baselines (Mission-level Detection) ---")
for alg in ['Nominal_Planning', 'Greedy_Lookahead', 'Adaptive_GA']:
    print(f"vs {alg}: W={wins[alg]} L={losses[alg]} T={ties[alg]}")

print("\n--- Aggregate Results ---")
for alg in algos:
    dsr = metrics[alg]['detected'] / total_missions if total_missions > 0 else 0
    avg_energy = statistics.mean(metrics[alg]['energy']) if metrics[alg]['energy'] else 0
    avg_rmst = statistics.mean(metrics[alg]['rmst']) if metrics[alg]['rmst'] else 0
    avg_tick = statistics.mean(metrics[alg]['detection_tick']) if metrics[alg]['detection_tick'] else 0
    
    print(f"{alg}:")
    print(f"  DSR: {dsr:.3f}")
    print(f"  Avg Energy: {avg_energy:.2f}")
    print(f"  Avg RMST: {avg_rmst:.2f}")
    print(f"  Avg Detection Tick: {avg_tick:.2f}")

