# Graph Report - papers/  (2026-04-10)

## Corpus Check
- Corpus is ~2,788 words - fits in a single context window. You may not need a graph.

## Summary
- 107 nodes · 105 edges · 25 communities detected
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Pearl 2009 — Causality: Models, Reasoning, and Inference` - 5 edges
2. `Denning 1968 — Working Set Model for Program Behavior` - 5 edges
3. `nars_belief.rs` - 5 edges
4. `Lakshminarayanan 2017 — Predictive Uncertainty via Deep Ensembles` - 4 edges
5. `Gray & Reuter 1992 — Transaction Processing` - 4 edges
6. `Shin et al. 2012 — Mobile Application Usage Patterns` - 4 edges
7. `Uncertainty Quantification` - 4 edges
8. `Memory Management & Working Set` - 4 edges
9. `Prediction & Forecasting` - 4 edges
10. `Anomaly Detection` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Adams & MacKay 2007 — Bayesian Online Changepoint Detection` --conceptually_related_to--> `Uncertainty Quantification`  [EXTRACTED]
  papers/apollo_papers_corpus.md → papers/apollo_papers_corpus.md  _Bridges community 7 → community 4_
- `Denning 1968 — Working Set Model for Program Behavior` --references--> `chromium_manager.rs`  [EXTRACTED]
  papers/apollo_papers_corpus.md → papers/apollo_papers_corpus.md  _Bridges community 5 → community 2_
- `Denning 1968 — Working Set Model for Program Behavior` --references--> `temporal_predictor.rs`  [EXTRACTED]
  papers/apollo_papers_corpus.md → papers/apollo_papers_corpus.md  _Bridges community 5 → community 3_
- `Pei Wang 2013 — Non-Axiomatic Reasoning System (NARS)` --references--> `nars_belief.rs`  [EXTRACTED]
  papers/apollo_papers_corpus.md → papers/apollo_papers_corpus.md  _Bridges community 8 → community 7_
- `Yerkes & Dodson 1908 — Arousal and Performance Law` --conceptually_related_to--> `Affective & Cognitive Computing`  [EXTRACTED]
  papers/apollo_papers_corpus.md → papers/apollo_papers_corpus.md  _Bridges community 2 → community 7_

## Hyperedges (group relationships)
- **Causal Reasoning Papers** — corpus_pearl2009, corpus_granger1969, corpus_rubin1974, corpus_lamport1978 [EXTRACTED 1.00]
- **Uncertainty Quantification Papers** — corpus_lakshminarayanan2017, corpus_guo2017, corpus_jaynes2003, corpus_adams2007, corpus_murphy2012 [EXTRACTED 1.00]
- **Anomaly Detection Papers** — corpus_chandola2009, corpus_page1954, corpus_ramsauer2020, corpus_bricken2023, corpus_templeton2024 [EXTRACTED 1.00]
- **Reliability & Systems Papers** — corpus_gray1992, corpus_nygard2018, corpus_kleppmann2017, corpus_lampson1974 [EXTRACTED 1.00]
- **Memory Management & Working Set Papers** — corpus_denning1968, corpus_damos2023, corpus_ileakage2023, corpus_memtis2023, corpus_drepper2007, corpus_zipnn2024 [EXTRACTED 1.00]
- **Self-Improvement & Meta-Learning Papers** — corpus_yuan2024, corpus_doncieux2018, corpus_nichol2018, corpus_google_nl2025 [EXTRACTED 1.00]
- **Bounded Rationality Papers** — corpus_simon1955, corpus_peiwang2013, corpus_auer2002, corpus_russo2018, corpus_kuncheva2004 [EXTRACTED 1.00]
- **Prediction & Forecasting Papers** — corpus_welch2006, corpus_holtwinters, corpus_shin2012, corpus_huang2012, corpus_baezayates2015, corpus_tuli2022 [EXTRACTED 1.00]
- **Affective & Cognitive Science Papers** — corpus_mcgaugh2004, corpus_yerkes1908, corpus_bliss1973, corpus_occ1988, corpus_bhatt2024, corpus_peiwang2013 [EXTRACTED 1.00]
- **Deep Learning & Neural Architecture Papers** — corpus_ramsauer2020, corpus_bricken2023, corpus_templeton2024, corpus_dettmers2022, corpus_zerveas2021, corpus_vaswani2017, corpus_tuli2022 [INFERRED 0.95]

## Communities

### Community 0 - "Anomaly Detection & Signal Intelligence"
Cohesion: 0.16
Nodes (14): Anomaly Detection, Bricken et al. 2023 — Sparse Autoencoders Find Interpretable Features, Chandola et al. 2009 — Anomaly Detection: A Survey, Jain 1991 — Art of Computer Systems Performance Analysis, Lampson 1974 — One Policy Per Resource, Page 1954 — Continuous Inspection Schemes (CUSUM), Ramsauer et al. 2020 — Hopfield Networks is All You Need, Templeton et al. 2024 — Scaling Monosemanticity (+6 more)

### Community 1 - "Self-Improvement & Cognitive Architecture"
Cohesion: 0.15
Nodes (13): Self-Improvement / Self-Evaluation, Bhatt et al. 2024 — Dopamine Reward Prediction Error, Doncieux 2018 — Open-ended Learning Framework, Google Nested Learning 2025 — Multi-level Hierarchical Learning, Nichol 2018 — Reptile Meta-Learning, Schulman 2017 — Proximal Policy Optimization, Yuan 2024 — Self-Rewarding Language Models, cognitive_bus.rs (+5 more)

### Community 2 - "Systems Reliability & Data Management"
Cohesion: 0.25
Nodes (9): Reliability & Fault Tolerance, Gray & Reuter 1992 — Transaction Processing, Jones 2011 — Chromium Multi-Process Architecture, Kleppmann 2017 — Designing Data-Intensive Applications, Nygard 2018 — Release It! Bulkheading, Yerkes & Dodson 1908 — Arousal and Performance Law, chromium_manager.rs, journal.rs (+1 more)

### Community 3 - "Temporal Prediction & Forecasting"
Cohesion: 0.36
Nodes (8): Prediction & Forecasting, Baeza-Yates et al. 2015 — Predicting The Next App, Holt 1957 / Winters 1960 — Exponential Smoothing Methods, Huang et al. 2012 — Predicting Mobile App Usage via Context, Shin et al. 2012 — Mobile Application Usage Patterns, Welch & Bishop 2006 — Introduction to the Kalman Filter, holt_winters.rs, temporal_predictor.rs

### Community 4 - "Uncertainty Quantification & Ensembles"
Cohesion: 0.29
Nodes (8): Uncertainty Quantification, Guo 2017 — Calibration of Modern Neural Networks, Jaynes 2003 — Probability Theory: The Logic of Science, Kuncheva 2004 — Combining Pattern Classifiers, Lakshminarayanan 2017 — Predictive Uncertainty via Deep Ensembles, Murphy 2012 — Machine Learning: A Probabilistic Perspective, epistemic.rs, meta_cognition.rs

### Community 5 - "Memory Management & Working Set Theory"
Cohesion: 0.36
Nodes (8): Memory Management & Working Set, DAMOS 2023 — Data Access Monitor-based Operation Schemes, Denning 1968 — Working Set Model for Program Behavior, Drepper 2007 — What Every Programmer Should Know About Memory, iLeakage CCS 2023 — Speculative Execution Attacks on Apple Silicon, MEMTIS SOSP 2023 — Efficient Memory Tiering System, memory_analyzer.rs, window_sensor.rs

### Community 6 - "Causal Reasoning & Counterfactuals"
Cohesion: 0.43
Nodes (7): Causal Reasoning, Granger 1969 — Investigating Causal Relations by Econometric Models, Pearl 2009 — Causality: Models, Reasoning, and Inference, Rubin 1974 — Potential Outcomes Framework, causal_graph.rs, outcome_tracker.rs, planner.rs

### Community 7 - "Affective & Cognitive Neuroscience"
Cohesion: 0.38
Nodes (7): Affective & Cognitive Computing, Adams & MacKay 2007 — Bayesian Online Changepoint Detection, Bliss & Lømo 1973 — Long-term Potentiation, McGaugh 2004 — Amygdala and Memory Consolidation, Ortony, Clore & Collins 1988 — Cognitive Structure of Emotions (OCC), Pfau 2010 — Concept Drift in Data Streams, nars_belief.rs

### Community 8 - "Bounded Rationality & Bandit Strategies"
Cohesion: 0.47
Nodes (6): Bounded Rationality, Auer et al. 2002 — Finite-time Analysis of Multiarmed Bandit (UCB1), Pei Wang 2013 — Non-Axiomatic Reasoning System (NARS), Russo et al. 2018 — Thompson Sampling Tutorial, Simon 1955 — Behavioral Model of Rational Choice (Satisficing), effectiveness_tracker.rs

### Community 9 - "Transformer Architectures & Telemetry"
Cohesion: 1.0
Nodes (3): Vaswani et al. 2017 — Attention Is All You Need, Zerveas et al. 2021 — Transformer Time Series Representation Learning, telemetry_logger.rs

### Community 10 - "Adversarial Robustness"
Cohesion: 1.0
Nodes (2): Madry 2018 — Deep Learning Resistant to Adversarial Attacks, adversarial_probe.rs

### Community 11 - "Tail Latency & Execution"
Cohesion: 1.0
Nodes (2): Dean & Barroso 2013 — The Tail at Scale, execute_actions.rs

### Community 12 - "Display Power Management"
Cohesion: 1.0
Nodes (2): Chuang et al. 2013 — Display Power Management Policies, display_turbo.rs

### Community 13 - "Survival Analysis & Hazard Modeling"
Cohesion: 1.0
Nodes (2): Cox 1972 — Regression Models and Life Tables, hazard_model.rs

### Community 14 - "Human-Computer Interaction"
Cohesion: 1.0
Nodes (2): Card et al. 1983 — Psychology of Human-Computer Interaction, latency_monitor.rs

### Community 15 - "Reinforcement Learning"
Cohesion: 1.0
Nodes (2): Chen et al. 2002 — TOCTTOU Race Conditions, thermal_interrupt.rs

### Community 16 - "Policy Optimization"
Cohesion: 1.0
Nodes (2): Cao et al. 1994 — Application-Controlled File Caching, cache_warmer.rs

### Community 17 - "Sparse Autoencoders & Interpretability"
Cohesion: 1.0
Nodes (2): Dettmers et al. 2022 — LLM.int8() 8-bit Quantization, llm_inference_mode.rs

### Community 18 - "Process & Session Modeling"
Cohesion: 1.0
Nodes (2): Norris 1997 — Markov Chains, focus_markov.rs

### Community 19 - "Security & Race Conditions"
Cohesion: 1.0
Nodes (1): Nagle 1984 — Congestion Control in IP/TCP

### Community 20 - "Cache & Prefetch Strategies"
Cohesion: 1.0
Nodes (1): Amdahl 1967 — Single Processor Approach to Large Scale Computing

### Community 21 - "Concept Drift Detection"
Cohesion: 1.0
Nodes (1): Lamport 1978 — Time, Clocks, and Ordering of Events

### Community 22 - "Multi-Process Architecture"
Cohesion: 1.0
Nodes (1): Pirolli & Card 1999 — Information Foraging Theory

### Community 23 - "Nested Hierarchical Learning"
Cohesion: 1.0
Nodes (1): Tuli et al. 2022 — Forecasting CPU and Memory for Cloud Apps

### Community 24 - "Memory Compression & Tiering"
Cohesion: 1.0
Nodes (1): ZipNN arXiv:2411.05239 — Efficient Neural Network Compression

## Knowledge Gaps
- **46 isolated node(s):** `Madry 2018 — Deep Learning Resistant to Adversarial Attacks`, `Lampson 1974 — One Policy Per Resource`, `Jones 2011 — Chromium Multi-Process Architecture`, `Drepper 2007 — What Every Programmer Should Know About Memory`, `Nagle 1984 — Congestion Control in IP/TCP` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Adversarial Robustness`** (2 nodes): `Madry 2018 — Deep Learning Resistant to Adversarial Attacks`, `adversarial_probe.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tail Latency & Execution`** (2 nodes): `Dean & Barroso 2013 — The Tail at Scale`, `execute_actions.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Display Power Management`** (2 nodes): `Chuang et al. 2013 — Display Power Management Policies`, `display_turbo.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Survival Analysis & Hazard Modeling`** (2 nodes): `Cox 1972 — Regression Models and Life Tables`, `hazard_model.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Human-Computer Interaction`** (2 nodes): `Card et al. 1983 — Psychology of Human-Computer Interaction`, `latency_monitor.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reinforcement Learning`** (2 nodes): `Chen et al. 2002 — TOCTTOU Race Conditions`, `thermal_interrupt.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Policy Optimization`** (2 nodes): `Cao et al. 1994 — Application-Controlled File Caching`, `cache_warmer.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sparse Autoencoders & Interpretability`** (2 nodes): `Dettmers et al. 2022 — LLM.int8() 8-bit Quantization`, `llm_inference_mode.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Process & Session Modeling`** (2 nodes): `Norris 1997 — Markov Chains`, `focus_markov.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Security & Race Conditions`** (1 nodes): `Nagle 1984 — Congestion Control in IP/TCP`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cache & Prefetch Strategies`** (1 nodes): `Amdahl 1967 — Single Processor Approach to Large Scale Computing`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Concept Drift Detection`** (1 nodes): `Lamport 1978 — Time, Clocks, and Ordering of Events`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Multi-Process Architecture`** (1 nodes): `Pirolli & Card 1999 — Information Foraging Theory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Nested Hierarchical Learning`** (1 nodes): `Tuli et al. 2022 — Forecasting CPU and Memory for Cloud Apps`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Memory Compression & Tiering`** (1 nodes): `ZipNN arXiv:2411.05239 — Efficient Neural Network Compression`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `nars_belief.rs` connect `Affective & Cognitive Neuroscience` to `Bounded Rationality & Bandit Strategies`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `Affective & Cognitive Computing` connect `Affective & Cognitive Neuroscience` to `Systems Reliability & Data Management`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Pearl 2009 — Causality: Models, Reasoning, and Inference` (e.g. with `Granger 1969 — Investigating Causal Relations by Econometric Models` and `Rubin 1974 — Potential Outcomes Framework`) actually correct?**
  _`Pearl 2009 — Causality: Models, Reasoning, and Inference` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Lakshminarayanan 2017 — Predictive Uncertainty via Deep Ensembles` (e.g. with `Guo 2017 — Calibration of Modern Neural Networks` and `Kuncheva 2004 — Combining Pattern Classifiers`) actually correct?**
  _`Lakshminarayanan 2017 — Predictive Uncertainty via Deep Ensembles` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Madry 2018 — Deep Learning Resistant to Adversarial Attacks`, `Lampson 1974 — One Policy Per Resource`, `Jones 2011 — Chromium Multi-Process Architecture` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._