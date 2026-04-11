# Apollo Optimizer — Complete Research Corpus

All 56 papers, theses, and technical reports cited in the Apollo Optimizer codebase.
Organized by domain. Used as input for knowledge graph construction and NARS belief synthesis.

---

## 1. Machine Learning & Reinforcement Learning

### Yuan 2024 — Self-Rewarding Language Models
- **ID**: arXiv:2401.10020
- **Key idea**: Models can self-evaluate without external oracle. Iterative self-improvement loop.
- **Apollo use**: Causal graph as internal judge for decision quality. `self_reward.rs`, `cognitive_bus.rs`
- **Concept tags**: self-evaluation, internal-oracle, iterative-improvement, causal-judgment

### Schulman 2017 — Proximal Policy Optimization Algorithms
- **ID**: arXiv:1707.06347
- **Key idea**: Clipped surrogate objective for stable policy gradient updates. Trust-region without KL penalty overhead.
- **Apollo use**: PPO reward normalization for CognitiveRewardBus; tanh-style reward scaling. `cognitive_bus.rs`, main.rs:850
- **Concept tags**: policy-gradient, reward-normalization, stability, trust-region

### Nichol 2018 — On First-Order Meta-Learning Algorithms (Reptile)
- **ID**: arXiv:1803.02999
- **Key idea**: θ_slow global model + θ_fast per-task adaptation. First-order MAML approximation.
- **Apollo use**: Per-workload fingerprint meta-learning; fast adaptation to new process patterns. `reptile_meta.rs`
- **Concept tags**: meta-learning, fast-adaptation, per-workload, catastrophic-forgetting

### Adams & MacKay 2007 — Bayesian Online Changepoint Detection
- **ID**: arXiv:0710.3742
- **Key idea**: Online posterior over run-length; detects distributional shifts without labels.
- **Apollo use**: Early warning for regime change before threshold breach. `nars_belief.rs:589`
- **Concept tags**: changepoint-detection, online-learning, distributional-shift, early-warning

### Russo et al. 2018 — A Tutorial on Thompson Sampling
- **ID**: arXiv:1707.02038
- **Key idea**: Posterior sampling for exploration. Balances exploration/exploitation via uncertainty.
- **Apollo use**: Multi-armed bandit strategy for specialist selection. `effectiveness_tracker.rs:51`
- **Concept tags**: exploration-exploitation, bandit, uncertainty, posterior-sampling

### Madry 2018 — Towards Deep Learning Models Resistant to Adversarial Attacks
- **Key idea**: PGD adversarial training; robustness through worst-case perturbation.
- **Apollo use**: Adversarial self-testing under synthetic stress; AdversarialProbe module. `adversarial_probe.rs:14`
- **Concept tags**: robustness, adversarial-testing, worst-case, stress-testing

### Guo 2017 — On Calibration of Modern Neural Networks (ICML)
- **Key idea**: Temperature scaling for calibrated confidence. ECE as calibration metric.
- **Apollo use**: Metacognition module; calibrated second-order confidence. `meta_cognition.rs:13`
- **Concept tags**: calibration, confidence, temperature-scaling, metacognition

### Lakshminarayanan 2017 — Predictive Uncertainty via Deep Ensembles (NeurIPS)
- **Key idea**: Ensemble variance as epistemic uncertainty. Proper scoring rules.
- **Apollo use**: Epistemic uncertainty from specialist ensemble variance; predictive entropy. `epistemic.rs:19`
- **Concept tags**: epistemic-uncertainty, ensemble, variance, proper-scoring

### Pearl 2009 — Causality: Models, Reasoning, and Inference
- **Key idea**: do-calculus, structural causal models, counterfactual reasoning.
- **Apollo use**: Causal graph as decision oracle; counterfactual baseline for actions. `causal_graph.rs`, `planner.rs`
- **Concept tags**: causality, do-calculus, counterfactual, structural-model

### Doncieux 2018 — Open-ended Learning: Conceptual Framework (Frontiers in Neurorobotics)
- **Key idea**: Intrinsic motivation + environmental diversity → open-ended competence growth.
- **Apollo use**: Unified Cognitive Health Score; systemic self-regulation across workloads. `cognitive_health.rs:17`
- **Concept tags**: open-ended-learning, intrinsic-motivation, self-regulation, competence

---

## 2. Memory Management & Working Set Theory

### Denning 1968 — The Working Set Model for Program Behavior (CACM 11:5)
- **Key idea**: Working set W(t,τ) = pages referenced in [t-τ, t]. Page fault rate ∝ resident set deficit.
- **Apollo use**: Memory pressure thresholds; idle working set identification; long-idle freeze path. `chromium_manager.rs`, `temporal_predictor.rs`, `window_sensor.rs`
- **Concept tags**: working-set, page-fault, memory-pressure, idle-detection, resident-set

---

## 3. Systems, Databases & Reliability

### Nygard 2018 — Release It! (Ch.5 Bulkheading)
- **Key idea**: Bulkhead pattern: partition resources so failure domains are isolated.
- **Apollo use**: Isolate competing workloads (Chromium renderers vs. rustc build). `chromium_manager.rs:195`
- **Concept tags**: bulkhead, fault-isolation, resource-partitioning, resilience

### Gray & Reuter 1992 — Transaction Processing: Concepts and Techniques
- **Key idea**: Write-ahead log, ARIES crash recovery, atomic state transitions.
- **Apollo use**: Write-ahead journal; crash recovery on daemon restart; atomic replace of state files. `journal.rs`, `learned_state.rs`
- **Concept tags**: write-ahead-log, crash-recovery, atomicity, WAL, ARIES

### Kleppmann 2017 — Designing Data-Intensive Applications (Ch.3, §9, §11)
- **Key idea**: Log compaction, append-only structures, stream processing, minimal state.
- **Apollo use**: Log size-triggered rotation; minimal state + observability. main.rs:335,1369
- **Concept tags**: log-compaction, append-only, stream-processing, minimal-state

### Lampson 1974 — One Policy Per Resource
- **Key idea**: Each resource should have exactly one controlling policy. Separation of mechanism/policy.
- **Apollo use**: Centralized safety utility; resource policy consistency. `decide_actions.rs:22`
- **Concept tags**: policy-mechanism-separation, centralized-control, resource-policy

### Jones 2011 — Chromium Multi-Process Architecture
- **Key idea**: Renderer/browser process isolation. GPU process separation.
- **Apollo use**: Process-aware memory management for Chromium tabs. `chromium_manager.rs:32`
- **Concept tags**: multi-process, renderer-isolation, process-separation, browser

### Drepper 2007 — What Every Programmer Should Know About Memory
- **Key idea**: Cache hierarchy effects, NUMA, TLB costs, zero-copy patterns.
- **Apollo use**: Zero-alloc byte write optimization in hot loop. evolve results:16,19
- **Concept tags**: cache, zero-alloc, hot-path, memory-hierarchy, TLB

### Nagle 1984 — Congestion Control in IP/TCP Internetworks
- **Key idea**: Nagle algorithm: coalesce small writes. Batching to reduce per-packet overhead.
- **Apollo use**: TCP socket buffering optimization; batch size tuning. main.rs:1369
- **Concept tags**: batching, coalescing, socket-buffering, throughput

### Dean & Barroso 2013 — The Tail at Scale (CACM)
- **Key idea**: Hedged requests, micro-partitioning, straggler mitigation for P99 latency.
- **Apollo use**: Resource pooling to reduce tail latency outliers. `execute_actions.rs:252`
- **Concept tags**: tail-latency, P99, hedged-requests, outlier-mitigation

### Amdahl 1967 — Single Processor Approach to Large Scale Computing
- **Key idea**: Speedup limited by sequential fraction. Identify bottleneck for optimization.
- **Apollo use**: Hot loop optimization; sequential bottleneck analysis. evolve results:17
- **Concept tags**: bottleneck, speedup, serial-fraction, optimization

### Lamport 1978 — Time, Clocks, and Ordering of Events
- **Key idea**: Happened-before relation; logical clocks; causal ordering without global time.
- **Apollo use**: Causal ordering of concurrent optimization events. evolve results:9
- **Concept tags**: causality, logical-clocks, ordering, concurrency, happened-before

---

## 4. Statistics & Signal Processing

### Jain 1991 — The Art of Computer Systems Performance Analysis
- **Key idea**: EMA, percentile analysis, steady-state measurement, system modeling.
- **Apollo use**: EMA composite scoring; P75 vs P95 analysis; steady-state detection. `fluidity.rs`, `intelligence_score.rs`
- **Concept tags**: EMA, percentile, steady-state, performance-measurement

### Page 1954 — Continuous Inspection Schemes
- **Key idea**: CUSUM chart: cumulative sum of deviations for detecting persistent shifts. Detection lag = h/(δ-k).
- **Apollo use**: Change detection in pressure signals; anomaly flagging. `intelligence_score.rs:882`, `signal_intelligence.rs:1541`
- **Concept tags**: CUSUM, change-detection, statistical-process-control, shift-detection

### Welch & Bishop 2006 — An Introduction to the Kalman Filter
- **Key idea**: Recursive state estimation with Riccati equation. Prediction-correction cycle.
- **Apollo use**: Kalman prediction for memory pressure; adaptive R tuning. main.rs:1161
- **Concept tags**: Kalman-filter, state-estimation, Riccati, prediction-correction

### Chandola et al. 2009 — Anomaly Detection: A Survey (ACM CSUR §3.1)
- **Key idea**: Statistical, proximity-based, and model-based anomaly detection taxonomy.
- **Apollo use**: EMA-MAD streaming anomaly detection; statistical thresholds. `process_baseline.rs`, `decide_actions.rs`
- **Concept tags**: anomaly-detection, EMA-MAD, streaming, statistical-threshold

### Cox 1972 — Regression Models and Life Tables
- **Key idea**: Proportional hazards model. Risk = baseline × exp(β·features).
- **Apollo use**: Process freeze/thaw hazard calibration; Cox regression for risk scoring. `hazard_model.rs:259`
- **Concept tags**: Cox-hazard, proportional-hazards, survival-analysis, risk-scoring

### Jaynes 2003 — Probability Theory: The Logic of Science
- **Key idea**: Bayesian probability as extended logic; maximum entropy principle.
- **Apollo use**: Entropy-based reasoning; information theory foundations for uncertainty. AIS baseline
- **Concept tags**: Bayesian, entropy, maximum-entropy, information-theory

### Holt 1957 / Winters 1960 — Exponential Smoothing Methods
- **Key idea**: Double/triple exponential smoothing; damped trend forecasting.
- **Apollo use**: Holt-Winters forecasting for periodic workload patterns. `holt_winters.rs`
- **Concept tags**: exponential-smoothing, trend-forecasting, seasonal-patterns, time-series

### Rubin 1974 — Potential Outcomes Framework
- **Key idea**: Counterfactual definition of causal effects. ATE via difference-in-differences.
- **Apollo use**: Counterfactual baseline for action evaluation in OutcomeTracker. `outcome_tracker.rs:371`
- **Concept tags**: counterfactual, causal-inference, ATE, difference-in-differences

### Granger 1969 — Investigating Causal Relations by Econometric Models
- **Key idea**: Granger causality: X causes Y if past X improves Y prediction beyond past Y alone.
- **Apollo use**: Delayed causation in causal graph; multi-horizon causal analysis. `causal_graph.rs`
- **Concept tags**: Granger-causality, delayed-causation, time-series-causality, prediction

---

## 5. Computer Architecture & Application Usage Prediction

### SeongJae Park — DAMOS: Data Access Monitor-based Operation Schemes (arXiv:2303.05919)
- **Key idea**: Access pattern monitoring with adaptive precision. Region-based operation schemes.
- **Apollo use**: Access-guided memory operations; adaptive WSS estimation. `memory_analyzer.rs:311`
- **Concept tags**: DAMOS, working-set, access-monitoring, adaptive-precision

### Shin et al. 2012 — Understanding and Prediction of Mobile Application Usage Patterns
- **Key idea**: Temporal patterns; time-of-day prediction (~80% accuracy). Markov chains.
- **Apollo use**: Temporal prediction of process launch; time-of-day optimization. `temporal_predictor.rs`
- **Concept tags**: temporal-prediction, usage-patterns, time-of-day, Markov

### Huang et al. 2012 — Predicting Mobile Application Usage Using Contextual Information
- **Key idea**: Context-aware prediction (location, time, activity). Sequential patterns.
- **Apollo use**: Sequential pattern prediction for app launches. `temporal_predictor.rs:13`
- **Concept tags**: contextual-prediction, sequential-patterns, application-usage

### Baeza-Yates et al. 2015 — Predicting The Next App That You Are Going To Use
- **Key idea**: Markov chain with context; top-3 accuracy ~85%.
- **Apollo use**: Next-app prediction for proactive memory pre-warming. `temporal_predictor.rs:17`
- **Concept tags**: next-app-prediction, Markov-context, top-k-accuracy, proactive

---

## 6. UI, Display & Energy

### Chuang et al. 2013 — Display Power Management Policies in Practice
- **Key idea**: Activity-sensitive display dimming. Power savings vs. user annoyance tradeoff.
- **Apollo use**: Display power optimization; turbo display mode. `display_turbo.rs:15`
- **Concept tags**: display-power, activity-sensing, power-management, user-experience

### Card et al. 1983 — The Psychology of Human-Computer Interaction
- **Key idea**: Human perception of latency; response time guidelines (100ms, 1s, 10s).
- **Apollo use**: Subjective perception thresholds; multiple problems feel worse than one. `latency_monitor.rs:112`
- **Concept tags**: latency-perception, HCI, response-time, user-perception

---

## 7. Networking & Control Systems

### Pirolli & Card 1999 — Information Foraging in Information Access Environments
- **Key idea**: Information scent; optimal foraging theory applied to UI navigation.
- **Apollo use**: Session phase detection; temporal workload characterization. main.rs:2975
- **Concept tags**: session-phases, foraging-theory, workload-characterization, temporal

### Chen et al. 2002 — TOCTTOU Race Conditions
- **Key idea**: Time-of-check/time-of-use vulnerability; PID reuse attacks.
- **Apollo use**: PID verification before acting; security race condition prevention. `thermal_interrupt.rs:808`
- **Concept tags**: TOCTTOU, PID-reuse, security, race-condition

### Cao et al. 1994 — Application-Controlled File Caching, Prefetching, and Disk Scheduling
- **Key idea**: App-controlled prefetch hints reduce I/O wait by ~50%.
- **Apollo use**: App-controlled prefetch for process startup. `cache_warmer.rs:20`
- **Concept tags**: prefetch, cache-warming, app-controlled, I/O-reduction

---

## 8. Deep Learning & Neural Architectures

### Ramsauer et al. 2020 — Hopfield Networks is All You Need
- **Key idea**: Modern Hopfield networks with exponential storage capacity. Energy-based associative memory.
- **Apollo use**: Evolved anomaly detection via associative memory patterns. `evolved_anomaly.rs`
- **Concept tags**: Hopfield, associative-memory, energy-based, exponential-capacity

### Bricken et al. 2023 — Sparse Autoencoders Find Interpretable Features in Language Models
- **Key idea**: TopK sparse autoencoder; superposition; interpretable latent features.
- **Apollo use**: Interpretable anomaly feature extraction. `evolved_anomaly.rs:9`
- **Concept tags**: sparse-autoencoder, interpretability, TopK, superposition

### Templeton et al. 2024 — Scaling Monosemanticity
- **Key idea**: Large-scale sparse autoencoders extract monosemantic features from transformer internals.
- **Apollo use**: Feature interpretability at scale; TopK > L1 for anomaly features. `evolved_anomaly.rs:29`
- **Concept tags**: monosemanticity, sparse-autoencoder-scaling, interpretability, features

### Dettmers et al. 2022 — LLM.int8(): 8-bit Matrix Multiplication
- **Key idea**: Mixed precision 8-bit quantization; outlier decomposition for stable inference.
- **Apollo use**: Memory bottleneck awareness in LLM inference optimization mode. `llm_inference_mode.rs:30`
- **Concept tags**: quantization, 8-bit, LLM-inference, memory-efficiency

### Zerveas et al. 2021 — Transformer-based Multivariate Time Series Representation Learning
- **Key idea**: Unsupervised pre-training on multivariate time series via masked patches.
- **Apollo use**: Multivariate anomaly detection on system metrics. `telemetry_logger.rs:32`
- **Concept tags**: time-series-transformer, multivariate, unsupervised, anomaly

### Vaswani et al. 2017 — Attention Is All You Need
- **Key idea**: Scaled dot-product attention; multi-head attention; positional encoding.
- **Apollo use**: Attention magnitude compensation in metric scoring. `telemetry_logger.rs:79`
- **Concept tags**: attention, transformer, multi-head, positional-encoding

### Tuli et al. 2022 — Forecasting CPU and Memory for Containerized Cloud Applications
- **Key idea**: LSTM + attention for resource forecasting in cloud containers.
- **Apollo use**: Anomaly context capture; pre-event telemetry dumps. main.rs:842,3357
- **Concept tags**: LSTM, resource-forecasting, pre-event-capture, telemetry

---

## 9. Cognitive Science & Non-Axiomatic Reasoning

### Pei Wang 2013 — Non-Axiomatic Reasoning System (NARS)
- **Key idea**: TruthValue(frequency, confidence); revision rule; bounded rationality under uncertainty.
- **Apollo use**: NARS belief system for freeze decisions; drift detection with Bayesian forgetting. `nars_belief.rs`
- **Concept tags**: NARS, truth-value, bounded-rationality, revision-rule, epistemic

### Murphy 2012 — Machine Learning: A Probabilistic Perspective
- **Key idea**: Bayesian inference, conjugate priors, EM algorithm, graphical models.
- **Apollo use**: Bayesian weights in OutcomeTracker; prior update for specialist credibility.
- **Concept tags**: Bayesian-inference, conjugate-prior, graphical-model, posterior-update

### Kuncheva 2004 — Combining Pattern Classifiers: Methods and Algorithms
- **Key idea**: Diversity in classifier ensembles; fusion rules; error-ambiguity decomposition.
- **Apollo use**: Specialist voting; ensemble combination for pressure decisions.
- **Concept tags**: ensemble, classifier-fusion, diversity, voting

### Pfau 2010 — (Concept Drift in data streams)
- **Key idea**: Detecting non-stationary distributions in streaming data.
- **Apollo use**: NARS concept drift detection; adaptive decay 0.95/persist cycle. `nars_belief.rs`
- **Concept tags**: concept-drift, non-stationary, streaming, adaptive-decay

### Simon 1955 — A Behavioral Model of Rational Choice
- **Key idea**: Satisficing: accept "good enough" solution given bounded rationality constraints.
- **Apollo use**: Satisficing confirmation signal for action decisions. main.rs:334
- **Concept tags**: satisficing, bounded-rationality, good-enough, decision-making

### Norris 1997 — Markov Chains (Cambridge University Press)
- **Key idea**: Irreducible, aperiodic chains; stationary distribution; mixing time.
- **Apollo use**: Process focus Markov chain; state transition modeling. `focus_markov.rs:8`
- **Concept tags**: Markov-chain, stationary-distribution, state-transitions, mixing-time

### Auer et al. 2002 — Finite-time Analysis of the Multiarmed Bandit Problem
- **Key idea**: UCB1 algorithm; O(√(nK log n)) regret bound; confidence bound exploration.
- **Apollo use**: UCB exploration for specialist selection; regret minimization. `effectiveness_tracker.rs:53`
- **Concept tags**: UCB, bandit, regret-bound, exploration-exploitation

---

## 10. Affective Computing & Neuroscience

### McGaugh 2004 — The Amygdala Modulates the Consolidation of Memories of Emotionally Arousing Experiences
- **Key idea**: Amygdala strengthens consolidation of emotionally arousing memories. Norepinephrine gate.
- **Apollo use**: Affective salience weighting; arousal-gated memory consolidation. `nars_belief.rs`
- **Concept tags**: affective-salience, memory-consolidation, arousal, emotional-memory

### Yerkes & Dodson 1908 — The Relation of Strength of Stimulus to Rapidity of Habit-Formation
- **Key idea**: Inverted-U relationship between arousal and performance (Yerkes-Dodson law).
- **Apollo use**: ArousalState EMA; optimal arousal zone for learning rate adjustment. `learned_state.rs`
- **Concept tags**: Yerkes-Dodson, inverted-U, arousal-performance, optimal-zone

### Bliss & Lømo 1973 — Long-term Potentiation of Synaptic Transmission in the Dentate Area
- **Key idea**: Persistent synaptic strengthening (LTP) from high-frequency stimulation.
- **Apollo use**: Long-term potentiation analog for belief confidence growth. `nars_belief.rs`
- **Concept tags**: LTP, synaptic-strengthening, long-term-potentiation, persistence

### Ortony, Clore & Collins 1988 — The Cognitive Structure of Emotions (OCC model)
- **Key idea**: Emotions as cognitive appraisals of events relative to goals/standards/attitudes.
- **Apollo use**: Cognitive appraisal of optimization outcomes; goal-relative evaluation.
- **Concept tags**: OCC-model, cognitive-appraisal, emotion-structure, goal-relative

### Bhatt et al. Nature Communications 2024 — Dopamine Reward Prediction Error
- **Key idea**: Dopaminergic RPE signals scale with surprise magnitude; temporal difference learning.
- **Apollo use**: Dopamine RPE analog for RL learning rate adjustment. `rl_threshold.rs:361`
- **Concept tags**: dopamine, RPE, reward-prediction-error, temporal-difference

---

## 11. Security & Privacy

### iLeakage CCS 2023 — Browser-based Speculative Execution Attacks on Apple Silicon
- **Key idea**: Speculative execution side-channel on M-series chips via Safari. Memory isolation required.
- **Apollo use**: Memory safety context for process isolation decisions. `memory_analyzer.rs`
- **Concept tags**: speculative-execution, side-channel, memory-isolation, security

### ZipNN arXiv:2411.05239 — Efficient Neural Network Compression
- **Key idea**: Near-lossless compression leveraging neural weight distributions.
- **Apollo use**: Memory compression efficiency context for swap management.
- **Concept tags**: compression, neural-weights, memory-efficiency, swap

### MEMTIS SOSP 2023 — Efficient Memory Tiering System
- **Key idea**: Hardware-assisted memory tiering; hot/warm/cold page classification.
- **Apollo use**: Memory tier classification; page temperature oracle. `memory_analyzer.rs`
- **Concept tags**: memory-tiering, page-temperature, hot-cold-classification, SOSP

---

## 12. Google Research & Industry

### Google Nested Learning 2025 — Multi-level Hierarchical Learning
- **Key idea**: L0 (fast reflex) / L1 (workload adaptation) / L2 (cross-workload generalization). Prevents catastrophic forgetting via hierarchy.
- **Apollo use**: NestedLearner L0/L1/L2 coordinator live in production. `nested_learner.rs`
- **Concept tags**: nested-learning, hierarchical, catastrophic-forgetting, multi-level

---

## Cross-Cutting Themes

| Theme | Papers |
|-------|--------|
| **Self-improvement** | Yuan 2024, Doncieux 2018, Nichol 2018 |
| **Causal reasoning** | Pearl 2009, Granger 1969, Rubin 1974 |
| **Uncertainty** | Lakshminarayanan 2017, Guo 2017, Jaynes 2003, Adams & MacKay 2007 |
| **Memory/Working Set** | Denning 1968, DAMOS, iLeakage, MEMTIS |
| **Reliability** | Gray & Reuter 1992, Nygard 2018, Kleppmann 2017 |
| **Bounded Rationality** | Simon 1955, Pei Wang 2013, Auer 2002 |
| **Prediction** | Kalman, Holt-Winters, Shin 2012, Baeza-Yates 2015 |
| **Affective/Cognitive** | McGaugh 2004, Yerkes-Dodson 1908, OCC 1988 |
| **Anomaly Detection** | Chandola 2009, Page 1954, Ramsauer 2020, Bricken 2023 |
