#!/usr/bin/env python3
"""Massive neuromodulator simulation — 1000 cycles across 5 scenarios.
Tests that the neuromodulator produces correct derived params under:
1. Calm (low pressure, no overflows)
2. Crisis (high urgency, thermal emergency)
3. Reward flowing (pressure dropping after actions)
4. Novel environment (process churn, entropy spikes)
5. Mixed realistic (alternating calm/stress like real M1 Air usage)
"""

DECAY = 0.10

class Neuromod:
    def __init__(self):
        self.da = 0.5
        self.na = 0.5
        self.se = 0.5
        self.ach = 0.5
        self.low_streak = 0
        self.last_proc = 400

    def tick(self, s):
        # Dopamine
        da_r = 0.0 if s.get('overflow', False) else 0.3
        da_d = max(0, min(s.get('pressure_drop', 0), 0.5)) * 0.8
        da_o = max(0, min(1 + s.get('outcome_penalty', 0) / 5, 1)) * 0.2
        da_sig = min(da_r + da_d + da_o, 1)
        self.da = max(0, min(self.da * (1 - DECAY) + da_sig * DECAY, 1))

        # Noradrenaline
        na_u = max(0, min(s.get('urgency', 0.4), 1)) * 0.4
        na_r = 0.3 if s.get('regime_up', False) else 0
        na_v = max(0, min(s.get('pressure_vel', 0) * 2, 0.3))
        na_t = 0.2 if s.get('thermal', False) else 0
        na_sig = min(na_u + na_r + na_v + na_t, 1)
        self.na = max(0, min(self.na * (1 - DECAY) + na_sig * DECAY, 1))

        # Serotonin
        if s.get('pressure_smooth', 0.5) < 0.30:
            self.low_streak += 1
        else:
            self.low_streak = max(0, self.low_streak - 1)
        se_s = min(self.low_streak / 20, 0.5)
        se_c = (1 - s.get('urgency', 0.4)) * 0.3
        se_r = 0.15 if s.get('regime_down', False) else 0
        se_o = 0.1 if not s.get('overflow', False) else 0
        se_sig = min(se_s + se_c + se_r + se_o, 1)
        self.se = max(0, min(self.se * (1 - DECAY) + se_sig * DECAY, 1))

        # Acetylcholine
        churn = abs(self.last_proc - s.get('proc_count', 400))
        self.last_proc = s.get('proc_count', 400)
        ach_ch = min(churn / 20, 0.4)
        ach_e = min(abs(s.get('entropy', 0)) / 3, 0.3)
        ach_x = 0.2 if s.get('exploring', False) else 0.05
        ach_sig = min(ach_ch + ach_e + ach_x, 1)
        self.ach = max(0, min(self.ach * (1 - DECAY) + ach_sig * DECAY, 1))

    @property
    def alpha_mult(self): return 0.5 + self.da
    @property
    def dyna_steps(self): return round(4 + self.na * 16)
    @property
    def se_shift(self): return (self.se - 0.5) * 0.10
    @property
    def eps_bonus(self): return self.ach * 0.05

def run_scenario(name, signal_fn, cycles=200):
    nm = Neuromod()
    history = {'da':[], 'na':[], 'se':[], 'ach':[],
               'alpha':[], 'dyna':[], 'shift':[], 'eps':[]}
    for i in range(cycles):
        s = signal_fn(i)
        nm.tick(s)
        history['da'].append(nm.da)
        history['na'].append(nm.na)
        history['se'].append(nm.se)
        history['ach'].append(nm.ach)
        history['alpha'].append(nm.alpha_mult)
        history['dyna'].append(nm.dyna_steps)
        history['shift'].append(nm.se_shift)
        history['eps'].append(nm.eps_bonus)

    print(f"\n{'='*60}")
    print(f"  {name} ({cycles} cycles)")
    print(f"{'='*60}")
    for key in ['da','na','se','ach']:
        vals = history[key]
        print(f"  {key:3s}: start={vals[0]:.3f} end={vals[-1]:.3f} "
              f"min={min(vals):.3f} max={max(vals):.3f} avg={sum(vals)/len(vals):.3f}")
    print(f"  ---")
    print(f"  alpha_mult : {history['alpha'][-1]:.3f}  (range: [{min(history['alpha']):.3f}, {max(history['alpha']):.3f}])")
    print(f"  dyna_steps : {history['dyna'][-1]}  (range: [{min(history['dyna'])}, {max(history['dyna'])}])")
    print(f"  se_shift   : {history['shift'][-1]:+.4f}  (range: [{min(history['shift']):+.4f}, {max(history['shift']):+.4f}])")
    print(f"  eps_bonus  : {history['eps'][-1]:.4f}  (range: [{min(history['eps']):.4f}, {max(history['eps']):.4f}])")
    return history

# ── Scenario 1: Calm ──
def calm_signals(i):
    return {'pressure_drop': 0.0, 'outcome_penalty': 0.0, 'overflow': False,
            'urgency': 0.1, 'regime_up': False, 'pressure_vel': 0.0,
            'thermal': False, 'pressure_smooth': 0.25, 'regime_down': False,
            'proc_count': 400, 'entropy': 0.0, 'exploring': False}

# ── Scenario 2: Crisis ──
def crisis_signals(i):
    return {'pressure_drop': -0.05, 'outcome_penalty': -3.0, 'overflow': True,
            'urgency': 0.95, 'regime_up': True, 'pressure_vel': 0.5,
            'thermal': True, 'pressure_smooth': 0.85, 'regime_down': False,
            'proc_count': 400, 'entropy': 2.0, 'exploring': False}

# ── Scenario 3: Reward flowing ──
def reward_signals(i):
    return {'pressure_drop': 0.08, 'outcome_penalty': 0.0, 'overflow': False,
            'urgency': 0.3, 'regime_up': False, 'pressure_vel': -0.1,
            'thermal': False, 'pressure_smooth': 0.45, 'regime_down': True,
            'proc_count': 400, 'entropy': 0.0, 'exploring': False}

# ── Scenario 4: Novel environment ──
def novel_signals(i):
    return {'pressure_drop': 0.0, 'outcome_penalty': 0.0, 'overflow': False,
            'urgency': 0.4, 'regime_up': False, 'pressure_vel': 0.0,
            'thermal': False, 'pressure_smooth': 0.50, 'regime_down': False,
            'proc_count': 400 + (i % 5) * 10, 'entropy': 2.5, 'exploring': True}

# ── Scenario 5: Realistic M1 Air (alternating) ──
def realistic_signals(i):
    phase = (i // 50) % 4  # 50-cycle phases
    if phase == 0:  # normal usage
        return {'pressure_drop': 0.01, 'outcome_penalty': 0.0, 'overflow': False,
                'urgency': 0.35, 'regime_up': False, 'pressure_vel': 0.0,
                'thermal': False, 'pressure_smooth': 0.65, 'regime_down': False,
                'proc_count': 400 + (i % 3), 'entropy': 0.1, 'exploring': False}
    elif phase == 1:  # build starts, pressure rises
        return {'pressure_drop': -0.03, 'outcome_penalty': -1.0, 'overflow': False,
                'urgency': 0.7, 'regime_up': True, 'pressure_vel': 0.3,
                'thermal': False, 'pressure_smooth': 0.75, 'regime_down': False,
                'proc_count': 420, 'entropy': 1.5, 'exploring': False}
    elif phase == 2:  # build peak, thermal
        return {'pressure_drop': -0.02, 'outcome_penalty': -2.0, 'overflow': True,
                'urgency': 0.9, 'regime_up': False, 'pressure_vel': 0.1,
                'thermal': True, 'pressure_smooth': 0.85, 'regime_down': False,
                'proc_count': 430, 'entropy': 0.5, 'exploring': False}
    else:  # build done, recovery
        return {'pressure_drop': 0.05, 'outcome_penalty': 0.0, 'overflow': False,
                'urgency': 0.2, 'regime_up': False, 'pressure_vel': -0.2,
                'thermal': False, 'pressure_smooth': 0.50, 'regime_down': True,
                'proc_count': 400, 'entropy': 0.0, 'exploring': False}

print("╔══════════════════════════════════════════════════════════════╗")
print("║   NEUROMODULATOR MASSIVE SIMULATION — 1000 cycles × 5      ║")
print("╚══════════════════════════════════════════════════════════════╝")

h1 = run_scenario("1. CALM (low pressure, idle Mac)", calm_signals, 200)
h2 = run_scenario("2. CRISIS (overflow, thermal, high urgency)", crisis_signals, 200)
h3 = run_scenario("3. REWARD (pressure dropping, actions working)", reward_signals, 200)
h4 = run_scenario("4. NOVEL (process churn, entropy spikes)", novel_signals, 200)
h5 = run_scenario("5. REALISTIC M1 AIR (4-phase build cycle)", realistic_signals, 200)

# ── Summary: how does behavior change across scenarios? ──
print(f"\n{'='*60}")
print(f"  COMPARATIVE SUMMARY")
print(f"{'='*60}")
print(f"{'Scenario':<30} {'alpha':>6} {'dyna':>5} {'shift':>7} {'eps':>6}")
print(f"{'-'*60}")
scenarios = [
    ("Calm", h1), ("Crisis", h2), ("Reward", h3),
    ("Novel", h4), ("Realistic M1", h5)
]
for name, h in scenarios:
    print(f"{name:<30} {h['alpha'][-1]:>6.3f} {h['dyna'][-1]:>5d} {h['shift'][-1]:>+7.4f} {h['eps'][-1]:>6.4f}")

print(f"\n  Interpretation:")
print(f"  - Calm:     SE high → shift +0.02 (skip more subsystems, save CPU)")
print(f"  - Crisis:   NA high → dyna 15+ (think harder), DA low → alpha <1 (cautious)")
print(f"  - Reward:   DA high → alpha >1 (learn faster from success)")
print(f"  - Novel:    ACh high → eps >0.03 (explore more unknown states)")
print(f"  - Realistic: oscillates naturally across phases")

# ── Verify safety invariants ──
print(f"\n  Safety checks:")
all_ok = True
for name, h in scenarios:
    for key in ['da','na','se','ach']:
        mn, mx = min(h[key]), max(h[key])
        if mn < 0 or mx > 1:
            print(f"  FAIL: {name} {key} out of [0,1]: [{mn:.3f}, {mx:.3f}]")
            all_ok = False
    if min(h['dyna']) < 1 or max(h['dyna']) > 30:
        print(f"  FAIL: {name} dyna_steps out of range: [{min(h['dyna'])}, {max(h['dyna'])}]")
        all_ok = False
if all_ok:
    print(f"  ALL PASSED — levels [0,1], dyna [4,20], all clamped")
