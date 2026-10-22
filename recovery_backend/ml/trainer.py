"""
Machine Learning Training Pipeline for Autonomic Fatigue Classification.

Adheres strictly to ML Best Practices:
- Strict Featurization Ordering: splits train/val/test BEFORE fitting any scalers
- Multi-Model Comparison: Regularized Logistic Regression, Random Forest, HistGradientBoosting, MLP
- Stratified 5-Fold Cross Validation
- Comprehensive evaluation: Macro F1, Precision, Recall, ROC-AUC, Confusion Matrix, Bootstrapped 95% CI
- Serialization of the winning model bundle to models/fatigue_classifier.joblib
"""

import json
import os
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ml.feature_extractor import BiometricFeatureExtractor
from ml.dataset_generator import DatasetGenerator
from ml.data_parser import UniversalDataParser


class FatigueModelTrainer:
    """
    Orchestrates data preparation, cross-validation, model comparison,
    and artifact export for the fatigue classification microservice.
    """

    CLASS_MAPPING = {0: "GREEN", 1: "YELLOW", 2: "RED"}

    def __init__(
        self,
        data_path: Optional[str] = None,
        models_dir: str = "models",
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.models_dir = models_dir
        self.random_state = random_state
        self.feature_names = BiometricFeatureExtractor.FEATURE_NAMES

    def load_or_generate_data(self) -> pd.DataFrame:
        """Load data from CSV/Excel/PDF if exists, otherwise generate synthetic dataset."""
        if self.data_path and os.path.exists(self.data_path):
            print(f"Loading empirical dataset from {self.data_path} using UniversalDataParser...")
            df = UniversalDataParser.parse_file(self.data_path, self.data_path)
            df = UniversalDataParser._standardize_dataframe(df)
        else:
            print("No external dataset provided. Generating synthetic physiological training cohort...")
            df = DatasetGenerator.generate_synthetic_dataset(
                n_samples_per_class=600,
                random_state=self.random_state,
                output_csv_path=self.data_path or "data/fatigue_dataset.csv",
            )

        # Validate schema & null check
        assert "label" in df.columns, "Dataset must contain 'label' column."
        df = df.dropna(subset=["label"] + self.feature_names)
        return df

    def train_and_evaluate(self) -> Dict[str, Any]:
        """
        Execute full training workflow:
        1. Train/Val/Test Split (70% / 15% / 15%)
        2. Fit StandardScaler strictly on Train split
        3. 5-Fold Stratified Cross-Validation on Candidate Models
        4. Validate on Validation Set & Compute Bootstrapped 95% CIs
        5. Evaluate Winner on Unseen Test Set
        6. Persist Model and Metadata
        """
        df = self.load_or_generate_data()

        X = df[self.feature_names].values
        y = df["label"].values.astype(int)

        # Step 1: Strict Featurization Ordering - Split before any scaling
        # 70% Train, 30% Temp (15% Val, 15% Test)
        X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=self.random_state, stratify=y
        )
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp_raw, y_temp, test_size=0.50, random_state=self.random_state, stratify=y_temp
        )

        print(f"Dataset Split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

        # Step 2: Fit scaler ONLY on X_train_raw
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        # Step 3: Define Candidate Models for Comparison
        candidates = {
            "LogisticRegression (Baseline)": LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=self.random_state,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=4,
                random_state=self.random_state,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=6,
                learning_rate=0.08,
                random_state=self.random_state,
            ),
            "MLP Neural Net": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.005,
                max_iter=600,
                random_state=self.random_state,
            ),
        }

        # Step 4: 5-Fold Cross Validation on Train Split
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_summary = {}

        for name, model in candidates.items():
            fold_f1s = []
            for train_idx, fold_val_idx in cv.split(X_train, y_train):
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_fval, y_fval = X_train[fold_val_idx], y_train[fold_val_idx]

                model.fit(X_tr, y_tr)
                preds = model.predict(X_fval)
                fold_f1s.append(f1_score(y_fval, preds, average="macro"))

            cv_summary[name] = {
                "cv_macro_f1_mean": float(np.mean(fold_f1s)),
                "cv_macro_f1_std": float(np.std(fold_f1s)),
            }
            print(f"[{name}] 5-Fold CV Macro F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")

        # Step 5: Fit on Full Training set and Evaluate on Validation set
        model_val_metrics = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            val_probs = model.predict_proba(X_val)

            macro_f1 = f1_score(y_val, val_preds, average="macro")
            weighted_f1 = f1_score(y_val, val_preds, average="weighted")
            acc = accuracy_score(y_val, val_preds)
            roc_auc = roc_auc_score(y_val, val_probs, multi_class="ovr")

            # 95% Confidence Interval for Macro F1 via Bootstrapping
            ci_lower, ci_upper = self._bootstrap_f1_ci(y_val, val_preds)

            model_val_metrics[name] = {
                "val_accuracy": round(float(acc), 4),
                "val_macro_f1": round(float(macro_f1), 4),
                "val_weighted_f1": round(float(weighted_f1), 4),
                "val_roc_auc": round(float(roc_auc), 4),
                "macro_f1_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
            }

        # Step 6: Select Winner based on Validation Macro F1
        winner_name = max(model_val_metrics.keys(), key=lambda k: model_val_metrics[k]["val_macro_f1"])
        print(f"\n[CHAMPION] Model Selected: {winner_name} (Val Macro F1 = {model_val_metrics[winner_name]['val_macro_f1']})")

        champion_model = candidates[winner_name]

        # Step 7: Final Retrain on Combined (Train + Val) and Final Test on Unseen Test Split
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        champion_model.fit(X_train_val, y_train_val)

        test_preds = champion_model.predict(X_test)
        test_probs = champion_model.predict_proba(X_test)

        test_macro_f1 = f1_score(y_test, test_preds, average="macro")
        test_accuracy = accuracy_score(y_test, test_preds)
        test_roc_auc = roc_auc_score(y_test, test_probs, multi_class="ovr")
        test_cm = confusion_matrix(y_test, test_preds).tolist()
        test_report = classification_report(
            y_test, test_preds, target_names=["GREEN", "YELLOW", "RED"], output_dict=True
        )

        test_metrics = {
            "test_accuracy": round(float(test_accuracy), 4),
            "test_macro_f1": round(float(test_macro_f1), 4),
            "test_roc_auc": round(float(test_roc_auc), 4),
            "confusion_matrix": test_cm,
            "classification_report": test_report,
        }
        print(f"Final Test Evaluation: Accuracy={test_accuracy:.4f}, Macro F1={test_macro_f1:.4f}, ROC-AUC={test_roc_auc:.4f}")

        # Feature Importance Analysis (if applicable)
        feature_importances = {}
        if hasattr(champion_model, "feature_importances_"):
            importances = champion_model.feature_importances_
            feature_importances = {
                self.feature_names[i]: round(float(importances[i]), 4)
                for i in np.argsort(-importances)
            }
        elif hasattr(champion_model, "coef_"):
            importances = np.mean(np.abs(champion_model.coef_), axis=0)
            feature_importances = {
                self.feature_names[i]: round(float(importances[i]), 4)
                for i in np.argsort(-importances)
            }

        # Step 8: Persist Model Bundle Artifacts
        os.makedirs(self.models_dir, exist_ok=True)
        model_bundle = {
            "model": champion_model,
            "scaler": scaler,
            "feature_names": self.feature_names,
            "class_mapping": self.CLASS_MAPPING,
            "winner_name": winner_name,
            "test_macro_f1": test_macro_f1,
        }

        model_path = os.path.join(self.models_dir, "fatigue_classifier.joblib")
        joblib.dump(model_bundle, model_path)
        print(f"Saved trained champion model bundle to {model_path}")

        # Save metadata report
        metadata = {
            "champion_model": winner_name,
            "features": self.feature_names,
            "class_mapping": self.CLASS_MAPPING,
            "cv_results": cv_summary,
            "validation_comparison": model_val_metrics,
            "test_metrics": test_metrics,
            "feature_importances": feature_importances,
            "dataset_size": len(df),
        }
        meta_path = os.path.join(self.models_dir, "training_report.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata and training report to {meta_path}")

        return metadata

    def _bootstrap_f1_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstraps: int = 1000,
    ) -> Tuple[float, float]:
        """Calculate 95% Confidence Interval for Macro F1 using bootstrapping."""
        rng = np.random.RandomState(self.random_state)
        bootstrapped_scores = []

        n = len(y_true)
        for _ in range(n_bootstraps):
            indices = rng.randint(0, n, n)
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = f1_score(y_true[indices], y_pred[indices], average="macro")
            bootstrapped_scores.append(score)

        if not bootstrapped_scores:
            return 0.0, 1.0

        lower = float(np.percentile(bootstrapped_scores, 2.5))
        upper = float(np.percentile(bootstrapped_scores, 97.5))
        return lower, upper


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Fatigue Recovery ML Classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to external CSV, Excel (.xlsx), Parquet, or PDF dataset.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save champion model bundle and report.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    trainer = FatigueModelTrainer(
        data_path=args.data_path,
        models_dir=args.models_dir,
        random_state=args.random_state,
    )
    trainer.train_and_evaluate()
