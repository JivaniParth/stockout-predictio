"""
Optuna Hyperparameter Tuning Module.
Defines per-model search spaces and executes objective evaluation on validation PR-AUC.
Supports Hybrid LSTM-Transformer, 1D-CNN, BiLSTM, Res-MLP, XGBoost, and Random Forest.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
import optuna
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader

from src.models.hybrid_transformer import HybridLSTMTransformer
from src.models.cnn import Tabular1DCNN
from src.models.lstm import BiLSTMStockoutPredictor
from src.models.dnn import DeepTabularMLP
from src.models.shared_trainer import SharedTrainer
from src.config import DEVICE


optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model_hyperparameters(
    model_name: str,
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_trials: int = 10,
    timeout_seconds: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes Optuna study maximizing validation PR-AUC for the target model architecture.
    """

    def objective(trial: optuna.Trial) -> float:
        if model_name == "HybridLSTMTransformer":
            embed_dim = trial.suggest_categorical("embed_dim", [32, 64])
            lstm_hidden = trial.suggest_categorical("lstm_hidden_dim", [16, 32])
            num_heads = trial.suggest_categorical("num_heads", [2, 4])
            dropout = trial.suggest_float("dropout_rate", 0.1, 0.3)
            lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)

            model = HybridLSTMTransformer(
                input_dim=input_dim,
                embed_dim=embed_dim,
                lstm_hidden_dim=lstm_hidden,
                num_heads=num_heads,
                dropout_rate=dropout,
            )
            trainer = SharedTrainer(
                model=model,
                model_name=f"Optuna_{model_name}",
                lr=lr,
                device=DEVICE,
                seed=seed,
            )
            res = trainer.fit(train_loader, val_loader, max_epochs=12, patience=4, verbose=False)
            return float(res["best_val_pr_auc"])

        elif model_name == "Tabular1DCNN":
            dropout = trial.suggest_float("dropout_rate", 0.1, 0.3)
            lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)

            model = Tabular1DCNN(input_dim=input_dim, dropout_rate=dropout)
            trainer = SharedTrainer(
                model=model,
                model_name=f"Optuna_{model_name}",
                lr=lr,
                device=DEVICE,
                seed=seed,
            )
            res = trainer.fit(train_loader, val_loader, max_epochs=12, patience=4, verbose=False)
            return float(res["best_val_pr_auc"])

        elif model_name == "BiLSTM":
            hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
            dropout = trial.suggest_float("dropout_rate", 0.1, 0.3)
            lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)

            model = BiLSTMStockoutPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout,
            )
            trainer = SharedTrainer(
                model=model,
                model_name=f"Optuna_{model_name}",
                lr=lr,
                device=DEVICE,
                seed=seed,
            )
            res = trainer.fit(train_loader, val_loader, max_epochs=12, patience=4, verbose=False)
            return float(res["best_val_pr_auc"])

        elif model_name == "DeepTabularMLP":
            hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
            dropout = trial.suggest_float("dropout_rate", 0.1, 0.3)
            lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)

            model = DeepTabularMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout,
            )
            trainer = SharedTrainer(
                model=model,
                model_name=f"Optuna_{model_name}",
                lr=lr,
                device=DEVICE,
                seed=seed,
            )
            res = trainer.fit(train_loader, val_loader, max_epochs=12, patience=4, verbose=False)
            return float(res["best_val_pr_auc"])

        elif model_name == "XGBoost":
            import xgboost as xgb
            max_depth = trial.suggest_int("max_depth", 3, 8)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 150)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)

            num_neg = (y_train == 0).sum()
            num_pos = (y_train == 1).sum()
            scale_pos = num_neg / max(num_pos, 1)

            clf = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample,
                scale_pos_weight=scale_pos,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:, 1]
            prec, rec, _ = precision_recall_curve(y_val, probs)
            return float(auc(rec, prec))

        elif model_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = trial.suggest_int("n_estimators", 50, 150)
            max_depth = trial.suggest_int("max_depth", 6, 14)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 8)

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            probs = rf.predict_proba(X_val)[:, 1]
            prec, rec, _ = precision_recall_curve(y_val, probs)
            return float(auc(rec, prec))

        else:
            raise ValueError(f"Unsupported model for tuning: {model_name}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    return {
        "model_name": model_name,
        "best_params": study.best_params,
        "best_pr_auc": study.best_value,
        "n_trials_completed": len(study.trials),
    }
