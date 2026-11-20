"""
Page 6: About the System, Methodology, and Academic Limitations Disclosures.
Fulfills all mandatory academic transparency disclosures.
"""

from pathlib import Path
import streamlit as st

st.header("ℹ️ About & Academic Limitations Disclosures")
st.markdown(
    "Comprehensive disclosure of methodology, empirical limitations, sample size constraints, "
    "and scientific caveats governing the Healthcare Supply Chain Management (HSCM) research project."
)

st.markdown(
    """
    ### 🔬 Research Project Context
    - **Academic Scope:** Master of Technology (M.Tech) in Computer Science & Engineering (Semester 3).
    - **Specialization Integration:** Cloud Computing (CC) & Deep Learning (DL) Applied Research.
    - **Student Investigator:** Parth Jivani (Enrolment No: 230303107002), Navrachana University.
    - **Faculty Supervisor:** Dr. Yogesh Chaudhari, School of Engineering and Technology.
    - **Repository Core:** `Implementation/` single-entry-point production architecture.
    """
)

st.markdown("---")
st.subheader("⚠️ Mandatory Academic Disclosures & Empirical Limitations")

st.markdown(
    """
    <div class='disclosure-box'>
    <h4>1. Contaminated Test Set Disclosure</h4>
    The prospective holdout partition (consisting of <strong>1,017 consignments scheduled for delivery in 2015+</strong>)
    was previously observed and iteratively evaluated during earlier exploratory research phases. Consequently,
    <strong>this test split is not a pristine, unaccessed holdout</strong>. Prospective metrics reported here represent
    confirmatory audit benchmarks rather than blind prospective validation.
    </div>

    <div class='disclosure-box'>
    <h4>2. Limited Positive Event Sample Size (N = 120)</h4>
    In the prospective 2015+ holdout partition of 1,017 shipments, there are exactly <strong>120 positive delay events</strong>
    (11.80% prevalence). With N=120 positive events, statistical power is constrained. 95% paired bootstrap confidence
    intervals for metric differences (PR-AUC, F1, Recall) frequently span zero when comparing top models, indicating that
    fine differences between leading architectures are <strong>statistically indistinguishable from random noise</strong>.
    </div>

    <div class='disclosure-box'>
    <h4>3. Post-Hoc Threshold Selection (Absence of True Pre-Registration)</h4>
    Decision thresholds (such as $\\tau_{\\text{safe}} = 0.08$ and $\\tau_{\\text{alert}} = 0.18$ for Policy A, and
    $\\tau_{\\text{safe}} = 0.30$ and $\\tau_{\\text{alert}} = 0.34$ for Policy B) were formulated via explicit algorithmic
    rules applied to 2014 validation data. However, these specific decision rules were selected <em>after exploratory data analysis</em>
    had revealed the underlying score distributions. They do not constitute truly pre-registered clinical operating thresholds.
    </div>

    <div class='disclosure-box'>
    <h4>4. Policy A Validation Alert Rate Budget Exceedance</h4>
    Policy A was nominally defined under a maximum operational alert budget of 30.0%. However, during out-of-fold validation
    calibration on 2014 data, Policy A selected an operating threshold yielding a <strong>validation alert rate of 31.9%</strong>,
    slightly exceeding its stated 30.0% ceiling before settling at 29.3% on the prospective holdout.
    </div>

    <div class='disclosure-box'>
    <h4>5. Gradient Boosted Trees Match or Outperform Deep Learning Architectures</h4>
    Empirical results demonstrate that on this tabular logistics dataset, classical gradient-boosted decision trees
    (<strong>RandomForest</strong> PR-AUC = 0.3353, ROC-AUC = 0.7949; <strong>XGBoost</strong> PR-AUC = 0.3599, ROC-AUC = 0.7861)
    consistently match or exceed the performance of complex deep learning architectures
    (<strong>BiLSTM</strong> PR-AUC = 0.2505, ROC-AUC = 0.7494; <strong>Deep Tabular MLP</strong> PR-AUC = 0.2606, ROC-AUC = 0.7694).
    Deep neural networks required substantial regularized inductive bias (Borderline-SMOTE, Focal Loss, AdamW) to remain competitive.
    </div>

    <div class='disclosure-box'>
    <h4>6. Exclusion of Delivery-Time Regressor Due to Negative R²</h4>
    The continuous delivery time regression head (investigated following Ghazvinian et al. 2025) yielded a
    <strong>negative coefficient of determination (R² = -0.063)</strong> on prospective holdout data once post-outcome date columns
    were strictly removed to eliminate temporal leakage. Because delivery durations in global public health logistics exhibit heavy
    unpredictable multimodal transit disruptions not explained by linear or neural feature embeddings, the delivery-time regressor
    is <strong>explicitly excluded as an operational production feature</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")
st.subheader("📚 Literature Citations & Scopus/WoS References")
st.markdown(
    """
    1. **Park, Y. et al. (2025).** *Smarter Chains, Safer Medicines: From Predictive Failures to Algorithmic Fixes in Global Pharmaceutical Logistics.* MDPI Forecasting, 7(1), 78.
    2. **Ghazvinian, A., Feng, B., & Feng, J. (2025).** *Enhancing Efficiency in the Healthcare Sector Through Multi-Objective Optimization of Freight Cost and Delivery Time in the HIV Drug Supply Chain Using Machine Learning.* MDPI Systems, 13(2), 91.
    3. **Yao, Y. (2010).** *Three-way decisions with probabilistic rough sets.* Information Sciences, 180(3), 341-353.
    4. **Zhang, C. et al. (2022).** *Research and application of XGBoost in imbalanced data.* International Journal of Distributed Sensor Networks, 18(9).
    5. **Cao, J. et al. (2024).** *Advanced hybrid LSTM-transformer architecture for real-time multi-task prediction in engineering systems.* Scientific Reports, 14, 18321.
    """
)
