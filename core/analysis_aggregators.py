"""
analysis_aggregators.py

Deterministic analysis that generates narrative "stories" and computed values suitable
for LLM consumption + template placeholder filling.

COMPLETE COVERAGE for all possible pharma analytics queries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class AnalysisAggregator:
    """Deterministic analysis with automatic column normalization"""

    def __init__(self, data: List[Dict], query_context: Dict):
        self.raw_data = data
        self.df = pd.DataFrame(data) if data else pd.DataFrame()
        self.context = query_context or {}
        self.analysis_results: Dict[str, Any] = {}
        
        # Normalize column names
        if not self.df.empty:
            self._normalize_columns()
    
    def _normalize_columns(self):
        """Normalize column names to expected format"""
        column_mappings = {
            # Common variations
            'name': 'model_name',
            'modelname': 'model_name',
            'model': 'model_name',
            'type': 'model_type',
            'modeltype': 'model_type',
            'metric': 'metric_name',
            'metricname': 'metric_name',
            'value': 'metric_value',
            'metricvalue': 'metric_value',
            'split': 'data_split',
            'datasplit': 'data_split',
            
            # Metric-specific columns (if in wide format)
            'test_rmse': 'rmse',
            'test_mae': 'mae',
            'test_r2': 'r2_score',
            'test_auc': 'auc_roc',
            'val_rmse': 'rmse',
            'val_mae': 'mae',
        }
        
        # Rename columns (case-insensitive)
        rename_dict = {}
        for col in self.df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            if col_lower in column_mappings:
                rename_dict[col] = column_mappings[col_lower]
        
        if rename_dict:
            self.df.rename(columns=rename_dict, inplace=True)
            print(f"[Aggregator] Normalized columns: {rename_dict}")

    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis dispatcher - called by analysis node after LLM decides which type.
        
        Returns a dict with:
            - analysis_type: str
            - story: str
            - computed_values: dict (flattened placeholders)
            - story_elements: dict
            - raw_metrics: dict (exact values for user display)
        """
        if self.df.empty:
            return self._empty_result()

        comparison_type = self.context.get("comparison_type", "general")
        use_case = self.context.get("use_case", "")

        # Route to appropriate aggregator
        if comparison_type == "performance":
            return self._analyze_model_performance()
        elif comparison_type == "drift":
            return self._analyze_drift()
        elif comparison_type == "ensemble_vs_base":
            return self._analyze_ensemble_vs_base()
        elif comparison_type == "feature_importance":
            return self._analyze_feature_importance()
        elif "uplift" in str(use_case).lower() or comparison_type == "uplift":
            return self._analyze_uplift()
        elif "territory" in str(use_case).lower():
            return self._analyze_territory_performance()
        elif "market" in str(use_case).lower():
            return self._analyze_market_share()
        elif "price" in str(use_case).lower():
            return self._analyze_price_sensitivity()
        elif "competitor" in str(use_case).lower():
            return self._analyze_competitor_share()
        elif "clustering" in str(use_case).lower():
            return self._analyze_clustering()
        elif comparison_type == "predictions":
            return self._analyze_predictions()
        elif comparison_type == "versions":
            return self._analyze_version_comparison()
        else:
            return self._analyze_general()

    # -------------------------
    # Utility: Standardizer
    # -------------------------
    def _standardize_placeholders(
        self,
        computed_values: Dict[str, Any],
        rankings: Optional[Dict[str, List]] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert complex computed values into a flat mapping of placeholders -> primitive values.
        Keys are UPPERCASE and safe for direct substitution in templates.
        """
        flat: Dict[str, Any] = {}

        # Timestamp
        flat["TIMESTAMP"] = computed_values.get("timestamp") or datetime.utcnow().isoformat()

        # Flatten metrics_summary into MODEL_METRIC -> mean
        if metrics_summary:
            models_seen = set()
            all_metrics = set()
            
            for metric, model_stats in metrics_summary.items():
                metric_key = str(metric).upper()
                all_metrics.add(metric_key)
                
                for model, stats in model_stats.items():
                    model_key = str(model).upper().replace(" ", "_").replace("-", "_")
                    models_seen.add(model_key)
                    
                    # e.g., RF_RMSE = mean value
                    try:
                        mean_val = float(stats.get("mean")) if isinstance(stats, dict) and "mean" in stats else float(stats)
                    except Exception:
                        mean_val = stats
                    flat[f"{model_key}_{metric_key}"] = mean_val

            flat["NUM_MODELS"] = len(models_seen)
            flat["NUM_METRICS"] = len(all_metrics)
            flat["MODELS_LIST"] = ", ".join(sorted(models_seen))

        # Flatten rankings: per-metric ordered list -> BEST_<METRIC>, SECOND_BEST_<METRIC>
        if rankings:
            winners = []
            second_best = []
            
            for metric, ranked in rankings.items():
                if ranked and len(ranked) > 0:
                    metric_key = str(metric).upper()
                    
                    # Best model
                    winner_name = str(ranked[0][0]).upper().replace(" ", "_").replace("-", "_")
                    winner_val = ranked[0][1]
                    flat[f"BEST_{metric_key}"] = winner_val
                    flat[f"BEST_{metric_key}_MODEL"] = winner_name
                    winners.append((metric_key, winner_name, winner_val))
                    
                    # Second best model
                    if len(ranked) >= 2:
                        second_name = str(ranked[1][0]).upper().replace(" ", "_").replace("-", "_")
                        second_val = ranked[1][1]
                        flat[f"SECOND_{metric_key}"] = second_val
                        flat[f"SECOND_{metric_key}_MODEL"] = second_name
                        second_best.append((metric_key, second_name, second_val))

            # Global BEST_MODEL from first metric ranking (fallback)
            if winners:
                first = winners[0]
                flat["BEST_MODEL"] = first[1]
                flat["BEST_METRIC_NAME"] = first[0]
                flat["BEST_METRIC_VALUE"] = first[2]
            
            if second_best:
                flat["SECOND_BEST_MODEL"] = second_best[0][1]
                flat["ALTERNATE_MODEL"] = second_best[0][1]

        # Drift summary
        if "drift_summary" in computed_values:
            drift_summary = computed_values.get("drift_summary", {}) or {}
            total_models = len(drift_summary)
            num_with_drift = sum(1 for v in drift_summary.values() if v.get("drift_detected"))
            flat["TOTAL_MODELS"] = total_models
            flat["NUM_WITH_DRIFT"] = num_with_drift
            flat["DRIFT_RATE"] = (num_with_drift / total_models * 100) if total_models > 0 else 0.0

        # Ensemble vs base - normalize advantage keys if present
        if "advantages" in computed_values:
            adv = computed_values.get("advantages", {}) or {}
            for metric, v in adv.items():
                metric_key = str(metric).upper()
                flat[f"ENSEMBLE_{metric_key}_PCT"] = v.get("percent_change")
                flat[f"ENSEMBLE_{metric_key}_ABS"] = v.get("absolute_diff")
                flat[f"ENSEMBLE_{metric_key}_BETTER"] = v.get("ensemble_better")

        # General numeric summaries
        if "numerical_summary" in computed_values:
            for col, stats in computed_values["numerical_summary"].items():
                col_key = str(col).upper()
                flat[f"{col_key}_MEAN"] = stats.get("mean")
                flat[f"{col_key}_STD"] = stats.get("std")

        # If there are top features, expose top feature names
        if "top_features" in computed_values and isinstance(computed_values["top_features"], dict):
            top_feats = list(computed_values["top_features"].keys())
            for i, feat in enumerate(top_feats[:5], 1):
                flat[f"TOP_FEATURE_{i}"] = feat
            flat["TOP_FEATURES_COUNT"] = len(top_feats)
        
        return flat

    # -------------------------
    # PERFORMANCE ANALYSIS
    # -------------------------
    def _analyze_model_performance(self) -> Dict[str, Any]:

        metrics_summary = self._compute_metrics_summary()
        
        # ===== EXTRACT MODEL LIST (CRITICAL) =====
        models_analyzed = []
        if "model_name" in self.df.columns:
            models_analyzed = sorted(self.df["model_name"].unique().tolist())
            print(f"[Performance Analysis] Found {len(models_analyzed)} models: {models_analyzed}")
        else:
            print(f"[Performance Analysis] WARNING: No model_name column!")
        
        rankings = self._compute_rankings(metrics_summary)
        story = self._generate_performance_story(metrics_summary, rankings)
        story_elements = self._extract_performance_story_elements(metrics_summary, rankings)
        computed_values = self._build_computed_values(metrics_summary, rankings)

        # Create raw_metrics for exact user display
        raw_metrics = self._extract_raw_metrics(metrics_summary, rankings)

        # Standardize/flatten
        flat = self._standardize_placeholders(computed_values, rankings=rankings, metrics_summary=metrics_summary)
        merged = {**computed_values, **flat}

        return {
            "analysis_type": "performance_comparison",
            "story": story,
            "story_elements": story_elements,
            "computed_values": merged,
            "raw_metrics": raw_metrics,
            "models_analyzed": models_analyzed,  # ← CRITICAL: Always include this
        }

    def _compute_metrics_summary(self) -> Dict[str, Any]:
        """
        FIXED VERSION - Properly handles wide-format SQL results
        
        Compute metrics summary - handles both LONG and WIDE formats
        
        Formats supported:
        1. LONG FORMAT: columns = [model_name, metric_name, metric_value]
        2. WIDE FORMAT: columns = [model_name, avg_test_rmse, avg_test_r2, ...]
        """
        metrics_summary = {}
        
        print(f"\n{'='*60}")
        print(f"[MetricsSummary] Starting computation")
        print(f"{'='*60}")
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Column dtypes:\n{self.df.dtypes}")
        
        if len(self.df) > 0:
            print(f"\n[Sample Row]:")
            print(self.df.iloc[0].to_dict())
        
        # Check if we have model_name column
        if 'model_name' not in self.df.columns:
            print("[MetricsSummary] ERROR: No 'model_name' column found!")
            return {}
        
        # ===== FORMAT DETECTION =====
        is_long_format = "metric_name" in self.df.columns and "metric_value" in self.df.columns
        
        if is_long_format:
            print("\n[MetricsSummary] Detected LONG format (metric_name + metric_value)")
            return self._compute_metrics_long_format()
        else:
            print("\n[MetricsSummary] Detected WIDE format (separate metric columns)")
            return self._compute_metrics_wide_format()


    def _compute_metrics_long_format(self) -> Dict[str, Any]:
        """Handle LONG format: [model_name, metric_name, metric_value]"""
        metrics_summary = {}
        
        for metric in self.df["metric_name"].unique():
            metric_df = self.df[self.df["metric_name"] == metric]
            metric_summary = {}
            
            for model in metric_df["model_name"].unique():
                values = metric_df[metric_df["model_name"] == model]["metric_value"].dropna()
                if len(values) > 0:
                    # Convert to float, handle any conversion errors
                    try:
                        values_float = values.astype(float)
                        metric_summary[model] = {
                            "mean": float(values_float.mean()),
                            "std": float(values_float.std()) if len(values_float) > 1 else 0.0,
                            "min": float(values_float.min()),
                            "max": float(values_float.max()),
                            "count": int(len(values_float)),
                        }
                        print(f"  [Long] {model} - {metric}: {metric_summary[model]['mean']:.4f}")
                    except Exception as e:
                        print(f"  [Long] ERROR converting {model} - {metric}: {e}")
                        continue
            
            if metric_summary:
                metrics_summary[metric] = metric_summary
        
        print(f"\n[Long Format] Found {len(metrics_summary)} metrics")
        return metrics_summary


    def _compute_metrics_wide_format(self) -> Dict[str, Any]:
        """
        ENHANCED VERSION - Handle WIDE format: [model_name, avg_test_rmse, avg_test_r2, ...]
        
        This is the format typically returned by SQL queries with aggregated metrics.
        """
        metrics_summary = {}
        
        # Identify non-metric columns
        non_metric_cols = [
            'model_name', 'model_type', 'model_id', 'execution_id', 
            'algorithm', 'version', 'use_case', 'data_split', 'created_at',
            'trained_date', 'prediction_date', 'status', 'description'
        ]
        
        # Find metric columns with improved detection
        metric_cols = []
        for col in self.df.columns:
            if col in non_metric_cols:
                continue
            
            # Get column data
            col_data = self.df[col]
            
            # Try to convert to numeric if it's object dtype
            if col_data.dtype == 'object':
                try:
                    col_data = pd.to_numeric(col_data, errors='coerce')
                except:
                    continue
            
            # Check if it's numeric and has non-null values
            if pd.api.types.is_numeric_dtype(col_data):
                # Check if it has any non-null numeric values
                if col_data.notna().sum() > 0:
                    metric_cols.append(col)
                    print(f"[Wide] Found metric column: {col} (dtype: {col_data.dtype}, non-null: {col_data.notna().sum()})")
        
        print(f"\n[Wide Format] Found {len(metric_cols)} metric columns: {metric_cols}")
        
        if not metric_cols:
            print("[Wide Format] ERROR: No numeric metric columns found!")
            print(f"Available columns: {list(self.df.columns)}")
            print(f"Sample data:\n{self.df.head(2)}")
            return {}
        
        # Process each metric column
        for metric_col in metric_cols:
            # Clean metric name: avg_test_rmse -> rmse
            metric_name = metric_col.lower()
            
            # Remove common prefixes
            for prefix in ['avg_', 'mean_', 'test_', 'val_', 'train_']:
                metric_name = metric_name.replace(prefix, '')
            
            # Remove common suffixes
            for suffix in ['_score', '_value', '_metric']:
                metric_name = metric_name.replace(suffix, '')
            
            print(f"\n[Wide] Processing {metric_col} as '{metric_name}'")
            
            metric_summary = {}
            
            # For wide format, each row is one model with all metrics
            for idx, row in self.df.iterrows():
                model = row["model_name"]
                value = row[metric_col]
                
                # Skip null/nan values
                if pd.isna(value):
                    print(f"  [Wide] Skipping {model}: {metric_col} is NULL")
                    continue
                
                # Convert to float
                try:
                    value = float(value)
                except (ValueError, TypeError) as e:
                    print(f"  [Wide] Skipping {model}: Cannot convert {value} to float ({e})")
                    continue
                
                # In wide format, we typically have one value per model
                # So mean = value (no aggregation needed across multiple rows)
                metric_summary[model] = {
                    "mean": value,
                    "std": 0.0,  # No std in wide format with single value per model
                    "min": value,
                    "max": value,
                    "count": 1,
                }
                
                print(f"  -> {model}: {metric_name} = {value:.6f}")
            
            if metric_summary:
                metrics_summary[metric_name] = metric_summary
                print(f"  [Wide] Stored metric '{metric_name}' with {len(metric_summary)} models")
            else:
                print(f"  [Wide] WARNING: No valid data for metric '{metric_name}'")
        
        # ===== FINAL VALIDATION =====
        print(f"\n{'='*60}")
        print(f"[MetricsSummary] FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total metrics found: {len(metrics_summary)}")
        
        for metric, models in metrics_summary.items():
            print(f"\n  Metric: {metric}")
            print(f"  Models: {len(models)}")
            
            # Show first model as example
            if models:
                first_model = list(models.keys())[0]
                first_value = models[first_model]
                print(f"  Example: {first_model} = {first_value['mean']:.6f}")
        
        if not metrics_summary:
            print("\n⚠ WARNING: No metrics extracted!")
            print("Possible issues:")
            print("  1. No numeric columns found")
            print("  2. All values are NULL")
            print("  3. Column names don't match expected patterns")
            print(f"\nDataFrame info:")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            print(f"  Dtypes:\n{self.df.dtypes}")
        
        print(f"{'='*60}\n")
        
        return metrics_summary

    def _compute_rankings(self, metrics_summary: Dict) -> Dict[str, List]:
        """Compute rankings for each metric"""
        rankings = {}
        lower_better = ["rmse", "mae", "mse", "mape"]

        for metric, model_values in metrics_summary.items():
            metric_lower = metric.lower()
            model_means = [(model, vals["mean"]) for model, vals in model_values.items()]
            if any(lb in metric_lower for lb in lower_better):
                ranked = sorted(model_means, key=lambda x: x[1])
            else:
                ranked = sorted(model_means, key=lambda x: x[1], reverse=True)
            rankings[metric] = ranked

        return rankings

    def _generate_performance_story(self, metrics_summary: Dict, rankings: Dict) -> str:
        """Generate narrative story for performance comparison"""
        story_parts: List[str] = []
        num_models = len(set(model for metric in metrics_summary.values() for model in metric))
        num_metrics = len(metrics_summary)
        story_parts.append(f"ANALYSIS_CONTEXT: Compared {num_models} models across {num_metrics} performance metrics.")

        for metric, model_values in metrics_summary.items():
            if metric not in rankings or len(rankings[metric]) < 1:
                continue
            ranked = rankings[metric]
            winner = ranked[0][0]
            winner_val = ranked[0][1]
            story_parts.append(f"\nMETRIC_STORY for {metric.upper()}:")
            story_parts.append(f"  - WINNER: {winner} ranked first with value {winner_val:.4f}")
            
            if len(ranked) >= 2:
                second = ranked[1][0]
                second_val = ranked[1][1]
                try:
                    gap_pct = abs((winner_val - second_val) / (second_val if second_val != 0 else 1) * 100)
                except Exception:
                    gap_pct = 0.0
                if gap_pct > 15:
                    story_parts.append(f"  - GAP_SIZE: LARGE - {winner} significantly outperforms {second} by over 15%")
                elif gap_pct > 5:
                    story_parts.append(f"  - GAP_SIZE: MODERATE - {winner} moderately outperforms {second} by 5-15%")
                else:
                    story_parts.append(f"  - GAP_SIZE: SMALL - {winner} and {second} are very close (under 5% difference)")

            winner_stats = model_values.get(winner, {})
            winner_std = winner_stats.get("std", 0)
            winner_mean = winner_stats.get("mean", 0)
            cv = (winner_std / winner_mean) if winner_mean != 0 else 0
            if cv < 0.1:
                story_parts.append(f"  - CONSISTENCY: {winner} shows HIGH consistency (very stable predictions)")
            elif cv < 0.3:
                story_parts.append(f"  - CONSISTENCY: {winner} shows MODERATE consistency")
            else:
                story_parts.append(f"  - CONSISTENCY: {winner} shows HIGH variability (unstable predictions)")

        return "\n".join(story_parts)

    def _extract_performance_story_elements(self, metrics_summary: Dict, rankings: Dict) -> Dict:
        """Extract structured story elements"""
        elements = {
            "winners_by_metric": {},
            "performance_gaps": {},
            "consistency_levels": {},
            "overall_winner": None,
            "decision_clarity": "unclear",
        }

        for metric, ranked in rankings.items():
            if len(ranked) >= 1:
                winner = ranked[0][0]
                elements["winners_by_metric"][metric] = winner
                if len(ranked) >= 2:
                    try:
                        gap = abs((ranked[0][1] - ranked[1][1]) / (ranked[1][1] if ranked[1][1] != 0 else 1) * 100)
                    except Exception:
                        gap = 0.0
                    elements["performance_gaps"][metric] = {
                        "winner": winner,
                        "runner_up": ranked[1][0],
                        "gap_category": "large" if gap > 15 else "moderate" if gap > 5 else "small",
                    }
            if winner in metrics_summary.get(metric, {}):
                winner_data = metrics_summary[metric][winner]
                cv = winner_data["std"] / winner_data["mean"] if winner_data["mean"] != 0 else 0
                elements["consistency_levels"][metric] = {
                    "model": winner,
                    "level": "high" if cv < 0.1 else "moderate" if cv < 0.3 else "low",
                }

        winners = list(elements["winners_by_metric"].values())
        if winners:
            counts = {m: winners.count(m) for m in set(winners)}
            overall_winner = max(counts, key=counts.get)
            dominance = counts[overall_winner] / len(winners)
            elements["overall_winner"] = {
                "model": overall_winner,
                "wins": counts[overall_winner],
                "total_metrics": len(winners),
                "dominance": "strong" if dominance >= 0.8 else "moderate" if dominance >= 0.5 else "weak",
            }
            elements["decision_clarity"] = "high" if dominance >= 0.8 else "moderate" if dominance >= 0.5 else "low"

        return elements

    def _build_computed_values(self, metrics_summary: Dict, rankings: Dict) -> Dict:
        """Build computed values dict"""
        return {"metrics_summary": metrics_summary, "rankings": rankings, "timestamp": datetime.utcnow().isoformat()}

    def _extract_raw_metrics(self, metrics_summary: Dict, rankings: Dict) -> Dict:
        """Extract exact metric values for user display (not for LLM)"""
        raw = {}
        
        for metric, model_stats in metrics_summary.items():
            raw[metric] = {}
            for model, stats in model_stats.items():
                raw[metric][model] = {
                    "mean": f"{stats['mean']:.6f}",
                    "std": f"{stats['std']:.6f}",
                    "min": f"{stats['min']:.6f}",
                    "max": f"{stats['max']:.6f}",
                    "count": stats['count']
                }
        
        return raw

    # -------------------------
    # DRIFT ANALYSIS
    # -------------------------
    def _analyze_drift(self) -> Dict[str, Any]:
        drift_summary = {}
        story_parts = []
        
        # Extract model list
        models_analyzed = []
        if "model_name" in self.df.columns:
            models_analyzed = sorted(self.df["model_name"].unique().tolist())
        
        if "model_name" in self.df.columns and "drift_detected" in self.df.columns:
            for model in self.df["model_name"].unique():
                model_df = self.df[self.df["model_name"] == model]
                drift_count = model_df["drift_detected"].sum()
                total_checks = len(model_df)
                drift_rate = (drift_count / total_checks * 100) if total_checks > 0 else 0
                
                avg_drift_score = model_df["drift_score"].mean() if "drift_score" in model_df.columns else 0
                
                drift_summary[model] = {
                    "drift_detected": drift_count > 0,
                    "drift_count": int(drift_count),
                    "total_checks": int(total_checks),
                    "drift_rate_pct": float(drift_rate),
                    "avg_drift_score": float(avg_drift_score)
                }
                
                if drift_count > 0:
                    story_parts.append(f"MODEL_DRIFT: {model} shows drift in {drift_count}/{total_checks} checks ({drift_rate:.1f}%)")
                    if drift_rate > 50:
                        story_parts.append(f"  - SEVERITY: HIGH - Frequent drift detected, retraining recommended")
                    elif drift_rate > 20:
                        story_parts.append(f"  - SEVERITY: MODERATE - Monitor closely")
                    else:
                        story_parts.append(f"  - SEVERITY: LOW - Acceptable drift levels")
        
        story = "\n".join(story_parts) if story_parts else "No significant drift detected across models"
        
        computed_values = {"drift_summary": drift_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "drift_details": {
                model: {
                    "drift_count": f"{stats['drift_count']}/{stats['total_checks']}",
                    "drift_rate": f"{stats['drift_rate_pct']:.2f}%",
                    "avg_score": f"{stats['avg_drift_score']:.6f}"
                }
                for model, stats in drift_summary.items()
            }
        }
        
        return {
            "analysis_type": "drift_analysis",
            "story": story,
            "story_elements": {"drift_summary": drift_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics,
            "models_analyzed": models_analyzed  # ← Add this
        }

    # -------------------------
    # ENSEMBLE VS BASE
    # -------------------------
    
    def _analyze_ensemble_vs_base(self) -> Dict[str, Any]:
        ensemble_metrics = {}
        base_metrics = {}
        
        # Extract model list
        models_analyzed = []
        if "model_name" in self.df.columns:
            models_analyzed = sorted(self.df["model_name"].unique().tolist())
        
        if "model_type" in self.df.columns and "model_name" in self.df.columns:
            ensemble_df = self.df[self.df["model_type"] == "ensemble"]
            base_df = self.df[self.df["model_type"] == "base_model"]
            
            # Get metric columns
            metric_cols = [col for col in self.df.columns if "metric" in col.lower() or any(m in col.lower() for m in ["rmse", "mae", "r2", "auc"])]
            
            for col in metric_cols:
                if col in ensemble_df.columns and col in base_df.columns:
                    ensemble_val = ensemble_df[col].mean()
                    base_val = base_df[col].mean()
                    
                    ensemble_metrics[col] = float(ensemble_val)
                    base_metrics[col] = float(base_val)
        
        # Compute advantages
        advantages = {}
        story_parts = ["ENSEMBLE VS BASE COMPARISON:"]
        
        for metric in ensemble_metrics.keys():
            ens_val = ensemble_metrics[metric]
            base_val = base_metrics.get(metric, 0)
            
            if base_val != 0:
                pct_change = ((ens_val - base_val) / abs(base_val)) * 100
                abs_diff = ens_val - base_val
                
                lower_better = any(lb in metric.lower() for lb in ["rmse", "mae", "mse", "mape"])
                ensemble_better = (abs_diff < 0) if lower_better else (abs_diff > 0)
                
                advantages[metric] = {
                    "ensemble_value": ens_val,
                    "base_value": base_val,
                    "percent_change": pct_change,
                    "absolute_diff": abs_diff,
                    "ensemble_better": ensemble_better
                }
                
                direction = "better" if ensemble_better else "worse"
                story_parts.append(f"  - {metric.upper()}: Ensemble is {abs(pct_change):.1f}% {direction} than base models")
        
        story = "\n".join(story_parts)
        
        computed_values = {
            "ensemble_metrics": ensemble_metrics,
            "base_metrics": base_metrics,
            "advantages": advantages,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "ensemble_exact": {k: f"{v:.6f}" for k, v in ensemble_metrics.items()},
            "base_exact": {k: f"{v:.6f}" for k, v in base_metrics.items()},
            "differences": {
                k: {
                    "absolute": f"{v['absolute_diff']:.6f}",
                    "percentage": f"{v['percent_change']:.2f}%",
                    "better": "Yes" if v['ensemble_better'] else "No"
                }
                for k, v in advantages.items()
            }
        }
        
        return {
            "analysis_type": "ensemble_vs_base",
            "story": story,
            "story_elements": {"advantages": advantages},
            "computed_values": merged,
            "raw_metrics": raw_metrics,
            "models_analyzed": models_analyzed  # ← Add this
        }

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        top_features = {}
        story_parts = ["FEATURE IMPORTANCE ANALYSIS:"]
        
        # Extract model list
        models_analyzed = []
        if "model_name" in self.df.columns:
            models_analyzed = sorted(self.df["model_name"].unique().tolist())
        
        if "feature_name" in self.df.columns and "importance_score" in self.df.columns:
            feature_groups = self.df.groupby("feature_name")["importance_score"].agg(['mean', 'std', 'count'])
            feature_groups = feature_groups.sort_values('mean', ascending=False)
            
            for i, (feature, row) in enumerate(feature_groups.head(10).iterrows(), 1):
                top_features[feature] = {
                    "rank": i,
                    "mean_importance": float(row['mean']),
                    "std_importance": float(row['std']),
                    "occurrences": int(row['count'])
                }
                
                story_parts.append(f"  {i}. {feature}: {row['mean']:.4f} (appears in {int(row['count'])} models)")
        
        story = "\n".join(story_parts)
        
        computed_values = {"top_features": top_features, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "features": {
                feat: {
                    "rank": stats['rank'],
                    "importance": f"{stats['mean_importance']:.6f}",
                    "std": f"{stats['std_importance']:.6f}",
                    "count": stats['occurrences']
                }
                for feat, stats in top_features.items()
            }
        }
        
        return {
            "analysis_type": "feature_importance",
            "story": story,
            "story_elements": {"top_features": top_features},
            "computed_values": merged,
            "raw_metrics": raw_metrics,
            "models_analyzed": models_analyzed  # ← Add this
        }

    # -------------------------
    # ADDITIONAL USE CASES
    # -------------------------
    
    def _analyze_uplift(self) -> Dict[str, Any]:
        """Analyze uplift modeling results"""
        uplift_summary = {}
        story_parts = ["UPLIFT ANALYSIS:"]
        
        if "uplift_score" in self.df.columns:
            avg_uplift = self.df["uplift_score"].mean()
            positive_uplift = (self.df["uplift_score"] > 0).sum()
            total = len(self.df)
            
            uplift_summary = {
                "avg_uplift_score": float(avg_uplift),
                "positive_uplift_count": int(positive_uplift),
                "positive_uplift_pct": float((positive_uplift / total * 100) if total > 0 else 0),
                "total_predictions": int(total)
            }
            
            story_parts.append(f"  - Average uplift score: {avg_uplift:.4f}")
            story_parts.append(f"  - {positive_uplift}/{total} predictions show positive uplift ({positive_uplift/total*100:.1f}%)")
            
            if "recommended_action" in self.df.columns:
                action_counts = self.df["recommended_action"].value_counts().to_dict()
                story_parts.append(f"  - Recommended actions: {action_counts}")
        
        story = "\n".join(story_parts)
        
        computed_values = {"uplift_summary": uplift_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "uplift_exact": {
                "avg_score": f"{uplift_summary.get('avg_uplift_score', 0):.6f}",
                "positive_rate": f"{uplift_summary.get('positive_uplift_pct', 0):.2f}%"
            }
        }
        
        return {
            "analysis_type": "uplift_analysis",
            "story": story,
            "story_elements": {"uplift_summary": uplift_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_territory_performance(self) -> Dict[str, Any]:
        """Analyze territory performance"""
        territory_summary = {}
        story_parts = ["TERRITORY PERFORMANCE ANALYSIS:"]
        
        if "entity_id" in self.df.columns and "prediction_value" in self.df.columns:
            for territory in self.df["entity_id"].unique():
                terr_df = self.df[self.df["entity_id"] == territory]
                
                territory_summary[territory] = {
                    "avg_predicted": float(terr_df["prediction_value"].mean()),
                    "avg_actual": float(terr_df["actual_value"].mean()) if "actual_value" in terr_df.columns else None,
                    "prediction_count": int(len(terr_df))
                }
            
            # Sort by performance
            sorted_terr = sorted(territory_summary.items(), 
                               key=lambda x: x[1]['avg_predicted'], 
                               reverse=True)
            
            story_parts.append(f"  - Top performing territories:")
            for terr, stats in sorted_terr[:5]:
                story_parts.append(f"    {terr}: ${stats['avg_predicted']:,.0f} predicted")
        
        story = "\n".join(story_parts)
        
        computed_values = {"territory_summary": territory_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "territories": {
                terr: {
                    "predicted": f"${stats['avg_predicted']:,.2f}",
                    "actual": f"${stats['avg_actual']:,.2f}" if stats['avg_actual'] else "N/A",
                    "count": stats['prediction_count']
                }
                for terr, stats in territory_summary.items()
            }
        }
        
        return {
            "analysis_type": "territory_performance",
            "story": story,
            "story_elements": {"territory_summary": territory_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_market_share(self) -> Dict[str, Any]:
        """Analyze market share predictions"""
        market_summary = {}
        story_parts = ["MARKET SHARE ANALYSIS:"]
        
        if "prediction_value" in self.df.columns:
            avg_share = self.df["prediction_value"].mean()
            max_share = self.df["prediction_value"].max()
            min_share = self.df["prediction_value"].min()
            
            market_summary = {
                "avg_market_share_pct": float(avg_share),
                "max_share_pct": float(max_share),
                "min_share_pct": float(min_share),
                "total_products": int(len(self.df))
            }
            
            story_parts.append(f"  - Average market share: {avg_share:.1f}%")
            story_parts.append(f"  - Range: {min_share:.1f}% to {max_share:.1f}%")
        
        story = "\n".join(story_parts)
        
        computed_values = {"market_summary": market_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "market_share_exact": {
                "average": f"{market_summary.get('avg_market_share_pct', 0):.4f}%",
                "maximum": f"{market_summary.get('max_share_pct', 0):.4f}%",
                "minimum": f"{market_summary.get('min_share_pct', 0):.4f}%"
            }
        }
        
        return {
            "analysis_type": "market_share",
            "story": story,
            "story_elements": {"market_summary": market_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_price_sensitivity(self) -> Dict[str, Any]:
        """Analyze price elasticity"""
        price_summary = {}
        story_parts = ["PRICE SENSITIVITY ANALYSIS:"]
        
        if "prediction_value" in self.df.columns:
            avg_elasticity = self.df["prediction_value"].mean()
            
            price_summary = {
                "avg_elasticity": float(avg_elasticity),
                "highly_sensitive": int((self.df["prediction_value"] < -1.5).sum()),
                "moderately_sensitive": int(((self.df["prediction_value"] >= -1.5) & (self.df["prediction_value"] < -0.5)).sum()),
                "inelastic": int((self.df["prediction_value"] >= -0.5).sum())
            }
            
            story_parts.append(f"  - Average price elasticity: {avg_elasticity:.2f}")
            story_parts.append(f"  - Highly sensitive segments: {price_summary['highly_sensitive']}")
        
        story = "\n".join(story_parts)
        
        computed_values = {"price_summary": price_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "elasticity": {
                "average": f"{price_summary.get('avg_elasticity', 0):.6f}",
                "distribution": {
                    "highly_sensitive": price_summary.get('highly_sensitive', 0),
                    "moderate": price_summary.get('moderately_sensitive', 0),
                    "inelastic": price_summary.get('inelastic', 0)
                }
            }
        }
        
        return {
            "analysis_type": "price_sensitivity",
            "story": story,
            "story_elements": {"price_summary": price_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_competitor_share(self) -> Dict[str, Any]:
        """Analyze competitor share forecasts"""
        comp_summary = {}
        story_parts = ["COMPETITOR SHARE ANALYSIS:"]
        
        if "entity_id" in self.df.columns and "prediction_value" in self.df.columns:
            for competitor in self.df["entity_id"].unique():
                comp_df = self.df[self.df["entity_id"] == competitor]
                
                comp_summary[competitor] = {
                    "avg_share": float(comp_df["prediction_value"].mean()),
                    "trend": "increasing" if comp_df["prediction_value"].iloc[-1] > comp_df["prediction_value"].iloc[0] else "decreasing"
                }
            
            sorted_comp = sorted(comp_summary.items(), key=lambda x: x[1]['avg_share'], reverse=True)
            
            story_parts.append(f"  - Top competitors by share:")
            for comp, stats in sorted_comp[:3]:
                story_parts.append(f"    {comp}: {stats['avg_share']:.1f}% ({stats['trend']})")
        
        story = "\n".join(story_parts)
        
        computed_values = {"competitor_summary": comp_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "competitors": {
                comp: {
                    "share": f"{stats['avg_share']:.4f}%",
                    "trend": stats['trend']
                }
                for comp, stats in comp_summary.items()
            }
        }
        
        return {
            "analysis_type": "competitor_share",
            "story": story,
            "story_elements": {"competitor_summary": comp_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_clustering(self) -> Dict[str, Any]:
        """Analyze clustering results"""
        cluster_summary = {}
        story_parts = ["CLUSTERING ANALYSIS:"]
        
        if "prediction_class" in self.df.columns:
            cluster_counts = self.df["prediction_class"].value_counts().to_dict()
            
            for cluster, count in cluster_counts.items():
                cluster_summary[cluster] = {
                    "count": int(count),
                    "percentage": float((count / len(self.df) * 100))
                }
                
                story_parts.append(f"  - {cluster}: {count} entities ({count/len(self.df)*100:.1f}%)")
        
        story = "\n".join(story_parts)
        
        computed_values = {"cluster_summary": cluster_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "clusters": {
                cluster: {
                    "count": stats['count'],
                    "percentage": f"{stats['percentage']:.2f}%"
                }
                for cluster, stats in cluster_summary.items()
            }
        }
        
        return {
            "analysis_type": "clustering",
            "story": story,
            "story_elements": {"cluster_summary": cluster_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_predictions(self) -> Dict[str, Any]:
        """Analyze general predictions"""
        pred_summary = {}
        story_parts = ["PREDICTIONS ANALYSIS:"]
        
        if "prediction_value" in self.df.columns:
            pred_summary = {
                "total_predictions": int(len(self.df)),
                "avg_prediction": float(self.df["prediction_value"].mean()),
                "std_prediction": float(self.df["prediction_value"].std())
            }
            
            if "actual_value" in self.df.columns:
                pred_summary["avg_actual"] = float(self.df["actual_value"].mean())
                pred_summary["avg_error"] = float((self.df["prediction_value"] - self.df["actual_value"]).abs().mean())
            
            story_parts.append(f"  - Total predictions: {pred_summary['total_predictions']}")
            story_parts.append(f"  - Average predicted value: {pred_summary['avg_prediction']:.2f}")
        
        story = "\n".join(story_parts)
        
        computed_values = {"prediction_summary": pred_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "predictions_exact": {
                k: f"{v:.6f}" if isinstance(v, float) else v
                for k, v in pred_summary.items()
            }
        }
        
        return {
            "analysis_type": "predictions",
            "story": story,
            "story_elements": {"prediction_summary": pred_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_version_comparison(self) -> Dict[str, Any]:
        """Analyze version comparisons"""
        version_summary = {}
        story_parts = ["VERSION COMPARISON ANALYSIS:"]
        
        if "old_version" in self.df.columns and "new_version" in self.df.columns:
            for idx, row in self.df.iterrows():
                model = row.get("model_name", f"model_{idx}")
                version_summary[model] = {
                    "old_version": row["old_version"],
                    "new_version": row["new_version"],
                    "verdict": row.get("performance_verdict", "unknown")
                }
                
                story_parts.append(f"  - {model}: {row['old_version']} → {row['new_version']} ({row.get('performance_verdict', 'unknown')})")
        
        story = "\n".join(story_parts)
        
        computed_values = {"version_summary": version_summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {"versions": version_summary}
        
        return {
            "analysis_type": "version_comparison",
            "story": story,
            "story_elements": {"version_summary": version_summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics
        }

    def _analyze_general(self) -> Dict[str, Any]:
        # Extract model list
        models_analyzed = []
        if "model_name" in self.df.columns:
            models_analyzed = sorted(self.df["model_name"].unique().tolist())
        
        summary = {
            "total_rows": int(len(self.df)),
            "columns": list(self.df.columns),
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns)
        }
        
        story_parts = ["GENERAL DATA ANALYSIS:"]
        story_parts.append(f"  - Total rows: {summary['total_rows']}")
        story_parts.append(f"  - Columns: {len(summary['columns'])}")
        
        if models_analyzed:
            story_parts.append(f"  - Models found: {len(models_analyzed)}")
            story_parts.append(f"  - Model list: {', '.join(models_analyzed[:5])}")
        
        if summary['numeric_columns']:
            for col in summary['numeric_columns'][:5]:
                summary[f"{col}_mean"] = float(self.df[col].mean())
                summary[f"{col}_std"] = float(self.df[col].std())
        
        story = "\n".join(story_parts)
        
        computed_values = {"general_summary": summary, "timestamp": datetime.utcnow().isoformat()}
        flat = self._standardize_placeholders(computed_values)
        merged = {**computed_values, **flat}
        
        raw_metrics = {
            "summary": {
                k: f"{v:.6f}" if isinstance(v, float) else v
                for k, v in summary.items()
            }
        }
        
        return {
            "analysis_type": "general",
            "story": story,
            "story_elements": {"summary": summary},
            "computed_values": merged,
            "raw_metrics": raw_metrics,
            "models_analyzed": models_analyzed  # ← Add this
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return result for empty data - ENHANCED"""
        return {
            "analysis_type": "empty",
            "story": "NO_DATA: No data available for analysis.",
            "story_elements": {},
            "computed_values": {"timestamp": datetime.utcnow().isoformat()},
            "raw_metrics": {},
            "models_analyzed": []  # ← Add this
        }

def analyze_data(data: List[Dict], query_context: Dict) -> Dict[str, Any]:
    """Main entry point for analysis"""
    aggregator = AnalysisAggregator(data, query_context)
    return aggregator.analyze()