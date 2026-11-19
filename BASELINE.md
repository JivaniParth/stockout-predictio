# Project Baseline Provenance & Cryptographic Verification

This document establishes the frozen baseline provenance for the USAID SCMS Delivery Delay Prediction project following repository restructuring into `Implementation`.

---

## 1. Frozen Baseline Predictions (`artifacts/baseline_predictions.json`)

```json
{
  "model_name": "XGBoost",
  "optimal_threshold": 0.35,
  "predictions_sha256": "a5d47cc0beea62f3126ada63d31dadb5db4593fea848656a70a4d882e4c7d2e8",
  "num_test_samples": 1017,
  "num_test_positives": 120,
  "test_probs_sample_first_10": [
    0.35028690099716187,
    0.06006342172622681,
    0.24005115032196045,
    0.04127925634384155,
    0.018168509006500244,
    0.07921189069747925,
    0.033696770668029785,
    0.014131426811218262,
    0.11332833766937256,
    0.0046686530113220215
  ],
  "git_commit": "9da48ab2afd8dd52a8c73d5c9f4b6c15965622ab"
}
```

---

## 2. Test Probabilities Cryptographic Hash Verification

- **Expected Target Array SHA-256**: `a5d47cc0beea62f3126ada63d31dadb5db4593fea848656a70a4d882e4c7d2e8`
- **Recomputed Array Bytes SHA-256 (`probs.tobytes()`)**: `a5d47cc0beea62f3126ada63d31dadb5db4593fea848656a70a4d882e4c7d2e8`
- **Verification Status**: **EXACT MATCH (Byte-for-byte preserved)**
- **Raw File on Disk SHA-256 (`baseline_test_probs.npy`)**: `c9d6b00a8169f68df3236755c229f12dc9181d8484e2fdfcc09114f6934526f2`

---

## 3. Imported Model and Artifact Cryptographic Hashes

### Serving & Export Artifacts (`artifacts/`)
| Artifact File | SHA-256 Checksum |
| :--- | :--- |
| `artifacts/baseline_predictions.json` | `446781ed3dab1c527aef59c9556c756db5dba2851533b62c59bd8687cd425760` |
| `artifacts/baseline_test_probs.npy` | `c9d6b00a8169f68df3236755c229f12dc9181d8484e2fdfcc09114f6934526f2` |
| `artifacts/champion.json` | `1ad6f07a1a65a43eb5ae443a38d3dabefe8a8fb02b70ff8a02ca6dc5fdac7942` |
| `artifacts/champion_model.onnx` | `fffcd4c3674e81f56a0b60c15d11f1f90546de5ccd1fe7b351903a40d0ce9b86` |
| `artifacts/manifest.json` | `0dd5ef63149111bf22803a09541ee2893e9e87aa5e41b04cb90d459855783995` |
| `artifacts/Tabular1DCNN.onnx` | `051d47ce7fa90fbdf492e26e8937f71841e109c65d258f6d82a110bf6957f195` |
| `artifacts/Tabular1DCNN.onnx.data` | `cd4c82b59c69f00cdfae3fda719a8180de023f9429c61271706557752923a5cf` |
| `artifacts/Tabular1DCNN.torchscript.pt` | `9c78566a17589eca80faaa3ff4fc391c881b1c36034a623dbf3be4f9e650c1a5` |
| `artifacts/XGBoost.onnx` | `c64b0973fb72aa7a5f0ccb0631b4b774c1a7d6f1eed85e3ca141cf2cf76aecdc` |

### Trained Model Checkpoints (`models/`)
| Model File | SHA-256 Checksum |
| :--- | :--- |
| `models/BiLSTM_seed_42_best.pt` | `9a328d734ea4243e7236916c6345dd6642fb9d00923c0738cb6ce97a4dd9b5a0` |
| `models/BiLSTM_seed_101_best.pt` | `62d46c37ea2ff6f811627aa8dff661adb350c682c9f175f731d495e26a709faa` |
| `models/BiLSTM_seed_202_best.pt` | `d0df75cacf5288cbcda2a0172e1e36e940dfa924ecc2d85302a25010a09be4ab` |
| `models/BiLSTM_seed_303_best.pt` | `fb45dfb6a39e0e20f2c1be638494125123d2decf32f7534df15705526ed7f46f` |
| `models/BiLSTM_seed_404_best.pt` | `741b8280a7a90c36c4a8be0337a118a5a317f094dc87e5f628e0b6b8ce45a29c` |
| `models/DeepTabularMLP_seed_42_best.pt` | `3b9adacc0d075c9f597521af692720333c2da68ee78f0ceebe23053f67b6a1c2` |
| `models/DeepTabularMLP_seed_101_best.pt` | `95a5b6cec97477295415333825d63194daf6d3aec6440e4131fd79b00fec5ba2` |
| `models/DeepTabularMLP_seed_202_best.pt` | `40e2ace132998a5c7e00d3dd8325b88cd6d66b45013a6922a593fd5c8493994d` |
| `models/DeepTabularMLP_seed_303_best.pt` | `833616de48924a14ed1dd28439f834255214a7675c41c94be3c3ab1c7eb947e3` |
| `models/DeepTabularMLP_seed_404_best.pt` | `9223f1dde32d29d45aed859ec85eb0d4056f314f79205a7e1b0621e9de405d44` |
| `models/HybridLSTMTransformer_seed_42_best.pt` | `2e98451b7f3f3a05d8a4d2b1c457d82b7818f2a9f43aff12dcb1e35771f5d890` |
| `models/HybridLSTMTransformer_seed_101_best.pt` | `f9ed7b54bf32466b3fb9064d3e6e3d03f768212f6f54c6ceea41a1aa754ed375` |
| `models/HybridLSTMTransformer_seed_202_best.pt` | `c713c4c82c4b1036b6cd8b3c34217048dbbae29ce6c719e79ab12210d8654952` |
| `models/HybridLSTMTransformer_seed_303_best.pt` | `850d640b6e97cebb676a1003ef99805c0cb17b9c0cc9b99cf13cd0af0c58f63a` |
| `models/HybridLSTMTransformer_seed_404_best.pt` | `d6206cede540007321bda30215efaf222ac564136f33013230f6c2b1cff250db` |
| `models/Tabular1DCNN_seed_42_best.pt` | `d4d8d0c7ff2630a033eba0e6680bf5434d55520853fb7b69eda6842bd88a99e8` |
| `models/Tabular1DCNN_seed_101_best.pt` | `ca8eb4d828a2b01ccf5bbbd4aa56e8233467be50fb8a5d3034ee7949a2e84cf4` |
| `models/Tabular1DCNN_seed_202_best.pt` | `c945fc41707301d67238485c33c6c839207d1924e2ae031988ba1d27883ea9ce` |
| `models/Tabular1DCNN_seed_303_best.pt` | `0dc9cfdb4be5811b1c2609f00dd9acff9013801d1788906154c266a3bf3c25d5` |
| `models/Tabular1DCNN_seed_404_best.pt` | `48b9cb7668bd8649b10480798d34f622e057062fc05f2891b71aa6d91e7b65aa` |
| `models/LogisticRegression.pkl` | `cfc099a3450c94cffdc81c5542b338ffc9144b829f584a450f39c71363660bc8` |
| `models/preprocessor.pkl` | `b6e51e42d667ee27a9c9aca8e9e16d569c8da7aa1d7586c58a30b26030aa5d72` |
| `models/RandomForest.pkl` | `7b0f52a8e3ebdfa83dd7c22b4daa9d9ffaca03d141f60fa0ed2b51a9e504e912` |
| `models/XGBoost.pkl` | `c560332984ca8e10c79152900e811f90d87ed0c226cd76112e8f62d69429503b` |

---

## 4. History Transition Note

> [!NOTE]
> The previous root repository git commit history (culminating in commit `496ffba`) and original `v1.0-baseline` tag have been abandoned following repository restructuring.
> This repository was re-initialized inside `Implementation`, with all historical baseline predictions, probabilities, and model binaries verified cryptographically against their recorded hashes.
> This snapshot represents tag `v1.0-baseline-imported`.
