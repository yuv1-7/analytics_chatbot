"""
analysis_aggregators.py

Deterministic analysis that generates narrative "stories" and computed values suitable
for LLM consumption + template placeholder filling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class AnalysisAggregator:
    """Deterministic analysis that generates narrative stories and standardized computed values."""

    def __init__(self, data: List[Dict], query_context: Dict):
        self.df = pd.DataFrame(data) if data else pd.DataFrame()
        self.context = query_context or {}
        self.analysis_results: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis dispatcher.
        Returns a dict with:
            - analysis_type: str
            - story: str
            - computed_values: dict (flattened placeholders)
            - story_elements: dict
            - any other metadata (e.g., models_analyzed)
        """
        if self.df.empty:
            return self._empty_result()

        comparison_type = self.context.get("comparison_type", "general")
        use_case = self.context.get("use_case", "")

        if comparison_type == "performance":
            return self._analyze_model_performance()
        elif comparison_type == "drift":
            return self._analyze_drift()
        elif comparison_type == "ensemble_vs_base":
            return self._analyze_ensemble_vs_base()
        elif comparison_type == "feature_importance":
            return self._analyze_feature_importance()
        elif "uplift" in str(use_case).lower():
            return self._analyze_uplift()
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
        rankings = self._compute_rankings(metrics_summary)
        story = self._generate_performance_story(metrics_summary, rankings)
        story_elements = self._extract_performance_story_elements(metrics_summary, rankings)
        computed_values = self._build_computed_values(metrics_summary, rankings)

        # Standardize/flatten
        flat = self._standardize_placeholders(computed_values, rankings=rankings, metrics_summary=metrics_summary)
        merged = {**computed_values, **flat}

        return {
            "analysis_type": "performance_comparison",
            "story": story,
            "story_elements": story_elements,
            "computed_values": merged,
            "models_analyzed": list(self.df["model_name"].unique()) if "model_name" in self.df.columns else [],
        }

    def _compute_metrics_summary(self) -> Dict[str, Any]:
        metrics_summary = {}

        if "metric_name" in self.df.columns:
            # long format
            for metric in self.df["metric_name"].unique():
                metric_df = self.df[self.df["metric_name"] == metric]
                if "model_name" in metric_df.columns:
                    metric_summary = {}
                    for model in metric_df["model_name"].unique():
                        values = metric_df[metric_df["model_name"] == model]["metric_value"].dropna()
                        if len(values) > 0:
                            metric_summary[model] = {
                                "mean": float(values.mean()),
                                "std": float(values.std()) if len(values) > 1 else 0.0,
                                "min": float(values.min()),
                                "max": float(values.max()),
                                "count": int(len(values)),
                            }
                    metrics_summary[metric] = metric_summary
        else:
            # wide format
            metric_cols = [col for col in self.df.columns if any(m in col.lower() for m in ["rmse", "mae", "r2", "auc", "accuracy"])]
            for metric_col in metric_cols:
                metric_name = metric_col.lower().replace("test_", "").replace("val_", "")
                if "model_name" in self.df.columns:
                    metric_summary = {}
                    for model in self.df["model_name"].unique():
                        values = self.df[self.df["model_name"] == model][metric_col].dropna()
                        if len(values) > 0:
                            metric_summary[model] = {
                                "mean": float(values.mean()),
                                "std": float(values.std()) if len(values) > 1 else 0.0,
                                "min": float(values.min()),
                                "max": float(values.max()),
                                "count": int(len(values)),
                            }
                    metrics_summary[metric_name] = metric_summary

        return metrics_summary

    def _compute_rankings(self, metrics_summary: Dict) -> Dict[str, List]:
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
            story_parts.append(f"  - WINNER: {winner} ranked first")
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

            if len(ranked) >= 3:
                last_val = ranked[-1][1]
                try:
                    range_pct = abs((winner_val - last_val) / (last_val if last_val != 0 else 1) * 100)
                except Exception:
                    range_pct = 0.0
                if range_pct > 30:
                    story_parts.append(f"  - SPREAD: WIDE - Performance varies greatly across models (>30% range)")
                elif range_pct > 10:
                    story_parts.append(f"  - SPREAD: MODERATE - Models show meaningful differences (10-30% range)")
                else:
                    story_parts.append(f"  - SPREAD: NARROW - All models perform similarly (<10% range)")

        if len(rankings) >= 2:
            winners = [ranked[0][0] for ranked in rankings.values() if ranked]
            if winners:
                winner_counts = {m: winners.count(m) for m in set(winners)}
                dominant_model = max(winner_counts, key=winner_counts.get)
                dominance = winner_counts[dominant_model] / len(rankings)
                story_parts.append(f"\nCROSS_METRIC_ANALYSIS:")
                if dominance >= 0.8:
                    story_parts.append(f"  - DOMINANCE: STRONG - {dominant_model} wins on most metrics (clear overall winner)")
                elif dominance >= 0.5:
                    story_parts.append(f"  - DOMINANCE: MODERATE - {dominant_model} wins on half the metrics (mixed results)")
                else:
                    story_parts.append(f"  - DOMINANCE: NONE - No single model dominates (highly context-dependent performance)")

        best_models = set(ranked[0][0] for ranked in rankings.values() if ranked)
        if len(best_models) == 1:
            story_parts.append(f"\nACTIONABILITY:")
            story_parts.append(f"  - CLARITY: HIGH - Clear single best model for deployment")
        else:
            story_parts.append(f"\nACTIONABILITY:")
            story_parts.append(f"  - CLARITY: LOW - Multiple models excel at different metrics, requires business prioritization")

        return "\n".join(story_parts)

    def _extract_performance_story_elements(self, metrics_summary: Dict, rankings: Dict) -> Dict:
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
        return {"metrics_summary": metrics_summary, "rankings": rankings, "timestamp": datetime.utcnow().isoformat()}

    # -------------------------
    # DRIFT ANALYSIS (stub - keep existing implementation)
    # -------------------------
    def _analyze_drift(self) -> Dict[str, Any]:
        # Keep your existing implementation
        return {"analysis_type": "drift_analysis", "story": "Drift analysis placeholder", "story_elements": {}, "computed_values": {}}

    # -------------------------
    # ENSEMBLE VS BASE (stub - keep existing implementation)
    # -------------------------
    def _analyze_ensemble_vs_base(self) -> Dict[str, Any]:
        # Keep your existing implementation
        return {"analysis_type": "ensemble_vs_base", "story": "Ensemble analysis placeholder", "story_elements": {}, "computed_values": {}}

    # -------------------------
    # FEATURE IMPORTANCE (stub - keep existing implementation)
    # -------------------------
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        # Keep your existing implementation
        return {"analysis_type": "feature_importance", "story": "Feature importance placeholder", "story_elements": {}, "computed_values": {}}

    # -------------------------
    # UPLIFT ANALYSIS (stub - keep existing implementation)
    # -------------------------
    def _analyze_uplift(self) -> Dict[str, Any]:
        # Keep your existing implementation
        return {"analysis_type": "uplift_analysis", "story": "Uplift placeholder", "story_elements": {}, "computed_values": {}}

    # -------------------------
    # GENERAL ANALYSIS (stub - keep existing implementation)
    # -------------------------
    def _analyze_general(self) -> Dict[str, Any]:
        # Keep your existing implementation
        return {"analysis_type": "general", "story": "General analysis placeholder", "story_elements": {}, "computed_values": {}}

    def _empty_result(self) -> Dict[str, Any]:
        return {"analysis_type": "empty", "story": "NO_DATA: No data available for analysis.", "story_elements": {}, "computed_values": {}}


def analyze_data(data: List[Dict], query_context: Dict) -> Dict[str, Any]:
    aggregator = AnalysisAggregator(data, query_context)
    return aggregator.analyze()