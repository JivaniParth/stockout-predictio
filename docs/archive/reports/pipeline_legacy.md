The current Deep Learning (DL) pipeline in this project is an end-to-end, production-grade intelligence system designed to predict
international pharmaceutical consignment delivery delays and impending clinical stockouts (specifically for Antiretrovirals/ARVs
and HIV/Rapid Diagnostic Tests across USAID/PEPFAR supply chains).

Unlike conventional offline academic models, this pipeline integrates mathematical deep sequence/attention modeling, leakage-safe
temporal validation, cloud microservice engineering (QoS), and algorithmic health equity governance.
──────

### High-Level Architecture of the Pipeline

│ Diagram exceeds terminal width (492 > 135 cols)
│ Displayed as code block. Widen terminal to view inline.

    flowchart TD
        subgraph DataIngestion["1. Data Ingestion & Leakage-Safe Splitting"]
            D1["SCMS Delivery History CSV\n(N=10,324 Shipments)"] --> D2["Temporal 3-Stage Split\n- Train: <= 2013\n- Val: 2014
            (Calibration)\n- Test: >= 2015 (Prospective)"]
        end

        subgraph FeatureEngineering["2. Preprocessing & Feature Engineering"]
            D2 --> F1["Regex Numeric Extraction & Flags\n(Weight, Freight, PO Lead Time)"]
            F1 --> F2["Leakage-Free Train Median Imputation"]
            F2 --> F3["Harmonic Date Encodings\n(sin/cos of Scheduled Month)"]
            F3 --> F4["SLA Lead Time Squeeze & Domain Ratios"]
            F4 --> F5["Categorical OHE & StandardScaler\n(Saved to preprocessor.pkl)"]
        end

        subgraph DeepLearningZoo["3. Deep Learning Models"]
            F5 --> M1["Hybrid LSTM-Transformer\n(Local Recurrence + Multi-Head Self-Attention)"]
            F5 --> M2["Tabular 1D-CNN\n(Edge Champion: 0.27ms, 0.28MB)"]
            F5 --> M3["Bidirectional LSTM\n(BiLSTM + Attention Pooling)"]
            F5 --> M4["Deep Tabular Res-MLP\n(LayerNorm + GELU + ResBlocks)"]
            F5 --> M5["Classical ML Baselines\n(XGBoost, Random Forest, Logistic Reg)"]
        end

        subgraph ImbalanceCalibration["4. Optimization & Decision Calibration"]
            M1 & M2 & M3 & M4 --> L1["Cost-Sensitive Weighted BCE Loss\n(Penalizes False Negatives)"]
            L1 --> L2["Threshold Calibration on 2014 Val\n(tau* = 0.21 maximizes F1 on delays)"]
        end

        subgraph TriEvaluation["5. Tri-Dimensional Co-Evaluation"]
            L2 --> E1["Predictive Accuracy\n(77.50% Delay Recall, PR-AUC, ROC-AUC)"]
            L2 --> E2["Cloud Infrastructure QoS\n(P50/P95 Latency, Batch Throughput, Memory)"]
            L2 --> E3["Algorithmic Health Equity\n(Tier 1 Hubs vs Tier 3 Fragile Corridors)"]
        end

        subgraph CloudServing["6. Cloud-Native Clean Serving"]
            E1 & E2 & E3 --> S1["FastAPI ASGI REST API & Streamlit UI"]
            S1 --> S2["Dynamic Micro-Batcher (Adaptive 8ms window)"]
            S1 --> S3["Tiered L1/L2 Cache with SingleFlight Locks"]
        end

──────

### 1. Data Ingestion & Leakage-Safe Chronological Partitioning

Implemented in Implementation/src/data/preprocessor.py using the USAID | PEPFAR Supply Chain Management System (SCMS) dataset (N =
10, 324 international consignments).

#### Ground Truth Binary Target

In public health logistics, delivery delays directly trigger buffer exhaustion and treatment interruptions. The binary target is
defined as:

         ⎧ 1  if Delivered to Client Dateᵢ > Scheduled Delivery Dateᵢ
    yᵢ = ⎩ 0  otherwise

Delayed shipments represent only ∼ 12% of the dataset, creating a severe class imbalance.

#### 3-Stage Forward Chronological Partition

Random K-fold CV or random train/test splits introduce severe temporal look-ahead leakage by training on future freight and vendor
trends to predict the past. The pipeline enforces strict chronological splitting:

1. Historical Training (≤ 2013, N = 7, 921): Used exclusively for parameter optimization, backpropagation, and fitting scaling
   parameters (μ, σ).
2. Validation & Calibration (2014, N = 1, 386): Used for early stopping and calibrating the decision threshold τ^\*.
3. Prospective Test (≥ 2015, N = 1, 017): Completely sequestered partition simulating live operational deployment on unseen future
   shipments.
   ──────

### 2. Preprocessing & Domain Feature Engineering

Implemented in HSCMPreprocessor (Implementation/src/data/preprocessor.py):

• Regex Extraction & Missingness Flags: Handles mixed text strings in columns like Weight (Kilograms) and Freight Cost (USD) (e.g.,
"Weight Captured Separately" or "Freight Included in Process"). Rather than dropping records, it parses numerical values and adds
explicit binary indicators: Weight_Missing_Flag, Freight_Included_Flag, PO_Date_Missing.
• Leakage-Free Imputation: Imputation medians (such as PO_Lead_Time_Days) are computed strictly on the historical train set and
frozen.
• Harmonic Cyclical Encodings: Projects periodic calendar delivery targets onto continuous trigonometric coordinates:
⎛2π · Month⎞ ⎛2π · Month⎞
Month = sin⎜──────────⎟, Month = cos⎜──────────⎟
sin ⎝ 12 ⎠ cos ⎝ 12 ⎠

• Manufacturer SLA Squeeze Dynamics: Measures lead-time compression between purchase order and delivery target:

    Δt    = (Scheduled Delivery Date - PO Date) - Manufacturer Lead Time
      SLA

A negative buffer indicates a manufacturer turnaround deficit, a prime indicator of impending delay.
• Domain Financial & Volumetric Ratios: Computes unit ratios with zero-division safety offsets:
Value Freight
Value_Per_Pack = ────────────, Freight_Per_Kg = ──────────
Quantity + 1 Weight + 1
• Log-Transformations & Categorical Grouping:
• Highly skewed numerical distributions (Line Item Quantity, Line Item Value, Weight) are stabilized using log(1 + x).
• High-cardinality vendors and manufacturing sites are frequency-bounded to top-15 entities, with rare entities collapsed into
"Other_Vendor" and "Other_Site".
• Categoricals are encoded via Scikit-Learn's OneHotEncoder(handle_unknown="ignore"), and numerics are normalized with
StandardScaler. The entire preprocessor is serialized to Implementation/results/models/preprocessor.pkl.

──────

### 3. Deep Learning Architecture Suite

The neural network implementations reside in Implementation/src/models/:

#### A. Hybrid LSTM-Transformer (Implementation/src/models/hybrid_transformer.py)

Combines local recurrent inductive bias with global multi-head self-attention:

1.  Tokenization: Projects flat tabular features into structured sequence tokens: ℝ^{D\_{in}} → ℝ^{B×L×D} (L = 6 pseudo-time steps, D
    = 64 embedding dim).
2.  BiLSTM Stage: Extracts local lead-time order dependencies across consecutive feature clusters.
3.  Transformer Encoder Block: 4-head multi-head self-attention (MHA) with residual connections, LayerNorm, and GELU feed-forward
    networks:

                                  ⎛ QKᵀ ⎞

    Attention(Q, K, V) = softmax⎜─────⎟V
    ⎝√(dₖ)⎠

4.  Attentive Softmax Pooling: Dynamically weights token importance into a single context vector:


    α = softmax(W₂ tanh(W₁Hᵀ)),   c = ∑ αᵢHᵢ

5. Output Head: Two-layer MLP with LayerNorm, GELU, and Sigmoid output.

#### B. Tabular 1D-CNN (Implementation/src/models/cnn.py) — The Edge Champion

• Treats normalized tabular feature vectors as a 1D spatial sequence: (B, 1, D\_{in}).
• Consists of two 1D convolution blocks with BatchNorm1d, ReLU, MaxPool1d, and AdaptiveAvgPool1d(16).
• Provides ultra-fast execution (0.27 ms inference), a tiny 0.28 MB memory footprint, and reaches the same top delay recall (77.
50%) as large sequence models.

#### C. Bidirectional LSTM (Implementation/src/models/lstm.py)

• 2-layer BiLSTM (D\_{hidden} = 64) over projected feature tokens with self-attentive pooling across temporal states.

#### D. Deep Tabular Residual MLP (Implementation/src/models/dnn.py)

• Deep feedforward architecture incorporating Tabular ResBlocks with skip-connections:

    y = GELU(x + ℱ(x))

• Employs LayerNorm and Dropout (0.2) to prevent overfitting on tabular feature correlations.
──────

### 4. Training Mechanics & Decision Calibration

Implemented in Implementation/src/models/trainer.py:

#### Rejection of SMOTE & Cost-Sensitive Weighted BCE

SMOTE was intentionally rejected based on healthcare literature showing it synthesizes clinically implausible, out-of-distribution
records. Instead, the trainer utilizes cost-sensitive weighted Binary Cross-Entropy loss:

           1   B
    ℒ = - ───  ∑  ⎡w    · yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ)⎤
           B  i=1 ⎣ pos                                    ⎦

where w*{pos} = min(N*{neg}/N\_{pos}, 6.0). This penalizes False Negatives (missed delayed shipments) significantly higher than
False Positives.

#### Threshold Calibration (τ^\* = 0.21)

Standard classification uses τ = 0.50, which is disastrous in clinical supply chains:

• False Positive (False Alarm): A logistics officer checks stock levels (minor administrative cost).
• False Negative (Missed Delay): The clinic runs completely out of ARVs; HIV patients stop therapy (clinical failure / death).

The pipeline sweeps thresholds across τ ∈ [0.10, 0.90] strictly on the 2014 Validation split, identifying τ^\* = 0.21 as the optimal
trade-off that maximizes minority F1 and captures over 77.50% of late shipments.
──────

### 5. Tri-Dimensional Co-Evaluation Benchmark

Results evaluated strictly on the Prospective 2015 Test Partition (N = 1, 017 unseen shipments) via
Implementation/run_experiments.py:

Model Architecture │ Test Acc │ Late Delay… │ F1-Score (… │ ROC-AUC │ PR-AUC │ P50 Latency… │ Throughput (BS=6… │ Disk Size
──────────────────────────┼──────────┼─────────────┼─────────────┼─────────┼────────┼──────────────┼───────────────────┼───────────
Hybrid LSTM-Transformer │ 70.70% │ 77.50% │ 0.3888 │ 0.7665 │ 0.2474 │ 0.46 ms │ 26,747 req/s │ 0.49 MB
Bidirectional LSTM │ 69.81% │ 77.50% │ 0.3804 │ 0.7758 │ 0.2866 │ 0.42 ms │ 29,508 req/s │ 0.90 MB
(BiLSTM) │ │ │ │ │ │ │ │
Tabular 1D-CNN (Edge │ 68.73% │ 77.50% │ 0.3705 │ 0.7404 │ 0.2668 │ 0.27 ms │ 45,235 req/s │ 0.28 MB
Optimal) │ │ │ │ │ │ │ │
Deep Tabular MLP │ 72.86% │ 75.00% │ 0.3947 │ 0.7397 │ 0.2426 │ 0.17 ms │ 70,299 req/s │ 0.96 MB
XGBoost (Ensemble) │ 75.12% │ 73.33% │ 0.4103 │ 0.7970 │ 0.3836 │ 0.33 ms │ 108,134 req/s │ 0.38 MB
Random Forest (Ensemble) │ 75.02% │ 70.83% │ 0.4009 │ 0.7877 │ 0.3226 │ 25.63 ms │ 2,440 req/s │ 5.75 MB
Logistic Regression │ 60.37% │ 82.50% │ 0.3295 │ 0.7271 │ 0.2374 │ 0.06 ms │ 947,615 req/s │ 0.002 MB
(Linear) │ │ │ │ │ │ │ │

• Layer 1: Domain Entities & Policies (Implementation/src/domain/entities.py, Implementation/src/domain/policies.py): Pure Python
data structures (ShipmentEntity, PredictionPrognosis, HealthEquityPolicy).
• Layer 2: Application Use Cases & Ports (Implementation/src/application/use_cases/predict_shipment.py): Orchestrates single and
batch predictions using abstract interfaces (IPredictor, IFeatureTransformer, ICacheService).
• Layer 3: Concrete Infrastructure Adapters:
• Dynamic Micro-Batcher (Implementation/src/infrastructure/adapters/dynamic_batcher.py): Solves the latency vs. throughput
trade-off by grouping concurrent single-item requests into micro-batches within an adaptive 8ms window.
• Tiered Cache with SingleFlight Locks (Implementation/src/infrastructure/cache/cache_service.py): Uses MD5 feature hashing to
return cached predictions in 0.001 ms, with mutex locking to prevent dog-piling / cache stampedes.
• Asynchronous Task Queue (Implementation/src/infrastructure/queue/task_queue.py): Background worker with Dead-Letter Queue
(DLQ) support for multi-thousand consignment manifests.
• Layer 4: Delivery Mechanisms:
• FastAPI Microservice (Implementation/src/deployment/app.py): REST endpoints (/predict, /batch, /telemetry, /health) with
token-bucket rate limiting and idempotency guards.
• Streamlit UI (Implementation/app_streamlit.py): Live clinician/logistics operator dashboard.
• Live Azure Deployment: Deployed on an Azure Standard_B1s VM (1 vCPU, 1 GB RAM, Ubuntu 24.04 LTS) fortified with 4.0 GB SSD
Swap space to absorb PyTorch memory spikes and prevent Linux Out-Of-Memory (OOM) crashes.

──────

### Key Summary of Pipeline Contributions

1. Zero Temporal Leakage: Replaces flawed random splits with a 3-stage forward chronological split (Train ≤ 2013, Val 2014,
   Prospective Test 2015).
2. Clinical Stockout Protection: Uses class-weighted BCE loss combined with validation threshold calibration (τ^\* = 0.21) to
   elevate delay detection recall to 77.50%.
3. Edge Deployment Champion: Discovered that the Tabular 1D-CNN matches the top 77.50% recall of larger Transformer models while
   consuming only 0.27 ms and 0.28 MB, making it ideal for low-cost clinics and edge nodes.
4. Cloud Infrastructure Hardening: Encapsulates DL inference in a clean hexagonal microservice with micro-batching, caching, rate
   limiting, and swap-protected Linux memory engineering.
