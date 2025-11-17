# advanced_story_generator.py
"""
AdvancedStoryGenerator
Deterministic, LLM-free multi-layer story generator for analytic pipelines.

This file provides a complete AdvancedStoryGenerator class with all previously
missing functions implemented. The implementations are deterministic, safe for
sensitive data contexts (do not emit raw rows), and produce qualitative
findings compatible with the rest of your pipeline.

Usage:
    from core.advanced_story_generator import AdvancedStoryGenerator
    story = AdvancedStoryGenerator(data, context).generate_comprehensive_story()
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime
from statistics import mean


@dataclass
class AnalyticalLayer:
    """Represents one layer of analysis"""
    layer_name: str
    findings: List[str]
    confidence: str  # HIGH, MEDIUM, LOW
    business_relevance: str  # CRITICAL, HIGH, MODERATE, LOW


class AdvancedStoryGenerator:
    """Generates multi-dimensional analytical stories"""

    def __init__(self, data: List[Dict], query_context: Dict):
        self.df = pd.DataFrame(data) if data else pd.DataFrame()
        self.context = query_context or {}
        self.layers: List[AnalyticalLayer] = []

    # -------------------------
    # Public entrypoint
    # -------------------------
    def generate_comprehensive_story(self) -> Dict[str, Any]:
        """Generate story with multiple analytical layers"""

        # Layer 1: Statistical Patterns (what happened)
        statistical_layer = self._analyze_statistical_patterns()

        # Layer 2: Comparative Insights (how models differ)
        comparative_layer = self._analyze_comparative_patterns()

        # Layer 3: Temporal Trends (changes over time)
        temporal_layer = self._analyze_temporal_patterns()

        # Layer 4: Anomaly Detection (unusual findings)
        anomaly_layer = self._detect_anomalies()

        # Layer 5: Business Context (why it matters)
        business_layer = self._interpret_business_context()

        # Layer 6: Decision Support (what to do)
        decision_layer = self._generate_decision_framework()

        # Synthesize into coherent narrative structure
        story_structure = self._synthesize_layers([
            statistical_layer,
            comparative_layer,
            temporal_layer,
            anomaly_layer,
            business_layer,
            decision_layer
        ])

        return story_structure

    # -------------------------
    # Layer implementations
    # -------------------------
    def _analyze_statistical_patterns(self) -> AnalyticalLayer:
        """Layer 1: Pure statistical patterns"""
        findings = []

        if self.df.empty:
            return AnalyticalLayer("STATISTICAL", ["NO_DATA"], "LOW", "LOW")

        # Distribution analysis on numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            try:
                col_series = self.df[col].dropna()
                if col_series.empty:
                    continue
                skewness = float(col_series.skew())
                kurtosis = float(col_series.kurtosis())

                if abs(skewness) > 1:
                    direction = "RIGHT" if skewness > 0 else "LEFT"
                    findings.append(
                        f"DISTRIBUTION_{col.upper()}: {direction}_SKEWED - "
                        f"Suggests {'a few high values' if skewness > 0 else 'a few low values'}"
                    )

                if kurtosis > 3:
                    findings.append(
                        f"DISTRIBUTION_{col.upper()}: HEAVY_TAILED - Potential outliers present"
                    )
            except Exception:
                continue

        # Correlation patterns (qualitative only)
        if len(numeric_cols) >= 2:
            correlations = self._detect_correlation_patterns(numeric_cols)
            findings.extend(correlations)

        confidence = "HIGH" if findings else "MEDIUM"
        relevance = "MODERATE" if findings else "LOW"

        return AnalyticalLayer(
            layer_name="STATISTICAL_PATTERNS",
            findings=findings if findings else ["NO_STRONG_STATISTICAL_PATTERNS_DETECTED"],
            confidence=confidence,
            business_relevance=relevance
        )

    def _analyze_comparative_patterns(self) -> AnalyticalLayer:
        """Layer 2: Model-to-model comparisons"""
        findings: List[str] = []

        if 'model_name' not in self.df.columns:
            return AnalyticalLayer("COMPARATIVE", ["NO_MODEL_COMPARISON"], "LOW", "LOW")

        models = self.df['model_name'].unique()

        if len(models) < 2:
            return AnalyticalLayer("COMPARATIVE", ["SINGLE_MODEL"], "LOW", "LOW")

        # Competitive landscape
        competitive_structure = self._assess_competitive_structure(models)
        findings.append(competitive_structure)

        # Performance gaps
        gap_analysis = self._analyze_performance_gaps()
        findings.extend(gap_analysis)

        # Cross-metric consistency
        consistency = self._analyze_cross_metric_consistency()
        findings.append(consistency)

        # Add simple model-level summary (counts)
        try:
            model_counts = int(self.df['model_name'].nunique())
            findings.append(f"MODELS_ANALYZED: {model_counts}")
        except Exception:
            pass

        return AnalyticalLayer(
            layer_name="COMPARATIVE_ANALYSIS",
            findings=findings,
            confidence="HIGH",
            business_relevance="CRITICAL"
        )

    def _analyze_temporal_patterns(self) -> AnalyticalLayer:
        """Layer 3: Time-based trends"""
        findings = []

        # Check for time columns
        time_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]

        if not time_cols:
            return AnalyticalLayer("TEMPORAL", ["NO_TEMPORAL_DATA"], "LOW", "LOW")

        time_col = time_cols[0]

        # Trend detection
        trends = self._detect_trends(time_col)
        findings.extend(trends)

        # Seasonality
        seasonality = self._detect_seasonality(time_col)
        if seasonality:
            findings.append(seasonality)

        # Momentum
        momentum = self._assess_momentum(time_col)
        if momentum:
            findings.append(momentum)

        confidence = "MEDIUM" if findings else "LOW"
        relevance = "HIGH" if any("TREND" in f or "SEASON" in f for f in findings) else "MODERATE"

        return AnalyticalLayer(
            layer_name="TEMPORAL_TRENDS",
            findings=findings if findings else ["NO_TEMPORAL_PATTERNS_DETECTED"],
            confidence=confidence,
            business_relevance=relevance
        )

    def _detect_anomalies(self) -> AnalyticalLayer:
        """Layer 4: Anomaly detection"""
        findings = []

        # Statistical outliers
        outliers = self._detect_statistical_outliers()
        if outliers:
            findings.extend(outliers)

        # Unexpected patterns
        unexpected = self._detect_unexpected_patterns()
        if unexpected:
            findings.extend(unexpected)

        # Data quality issues
        quality_issues = self._assess_data_quality()
        if quality_issues:
            findings.extend(quality_issues)

        confidence = "HIGH" if findings else "MEDIUM"
        relevance = "CRITICAL" if any("SEVERE" in f for f in findings) else "MODERATE"

        return AnalyticalLayer(
            layer_name="ANOMALY_DETECTION",
            findings=findings if findings else ["NO_ANOMALIES_DETECTED"],
            confidence=confidence,
            business_relevance=relevance
        )

    def _interpret_business_context(self) -> AnalyticalLayer:
        """Layer 5: Business interpretation"""
        findings: List[str] = []

        use_case = str(self.context.get('use_case', '')).lower()
        comparison_type = str(self.context.get('comparison_type', '')).lower()

        # Use case specific interpretations
        if 'nrx' in use_case or 'forecast' in use_case:
            findings.extend(self._interpret_forecasting_context())
        if 'hcp' in use_case or 'engagement' in use_case:
            findings.extend(self._interpret_hcp_context())
        if 'drift' in comparison_type:
            findings.extend(self._interpret_drift_context())
        if 'ensemble' in comparison_type:
            findings.extend(self._interpret_ensemble_context())

        # Generic business implications if none added
        if not findings:
            findings.append("BUSINESS_IMPACT: No specific business context identified; recommend confirming use_case.")

        return AnalyticalLayer(
            layer_name="BUSINESS_CONTEXT",
            findings=findings,
            confidence="MEDIUM",
            business_relevance="CRITICAL"
        )

    def _generate_decision_framework(self) -> AnalyticalLayer:
        """Layer 6: Decision support"""
        findings: List[str] = []

        # Assess decision clarity
        clarity = self._assess_decision_clarity()
        findings.append(clarity.get('description', 'DECISION_CLARITY: Unknown'))

        # Identify decision blockers
        blockers = self._identify_decision_blockers()
        if blockers:
            findings.extend(blockers)

        # Generate recommendation framework
        framework = self._build_recommendation_framework()
        if framework:
            findings.extend(framework)

        # Risk assessment
        risks = self._assess_deployment_risks()
        if risks:
            findings.extend(risks)

        return AnalyticalLayer(
            layer_name="DECISION_FRAMEWORK",
            findings=findings,
            confidence="HIGH" if clarity.get('score', 0) > 0.6 else "MEDIUM",
            business_relevance="CRITICAL"
        )

    # -------------------------
    # Synthesis helpers
    # -------------------------
    def _synthesize_layers(self, layers: List[AnalyticalLayer]) -> Dict[str, Any]:
        """Synthesize all layers into coherent story structure"""

        # Priority ranking for business_relevance
        priority_order = {
            "CRITICAL": 1,
            "HIGH": 2,
            "MODERATE": 3,
            "LOW": 4
        }

        # Sort layers by business relevance and confidence
        sorted_layers = sorted(
            layers,
            key=lambda x: (priority_order.get(x.business_relevance, 5), x.confidence),
        )

        narrative_structure = {
            "executive_summary": self._generate_executive_summary(sorted_layers),
            "key_findings": self._extract_key_findings(sorted_layers),
            "detailed_analysis": self._format_detailed_analysis(sorted_layers),
            "recommendations": self._consolidate_recommendations(sorted_layers),
            "confidence_assessment": self._assess_overall_confidence(sorted_layers),
            "metadata": {
                "layers_analyzed": len(layers),
                "high_confidence_findings": sum(1 for l in layers if l.confidence == "HIGH"),
                "critical_findings": sum(1 for l in layers if l.business_relevance == "CRITICAL"),
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        }

        return narrative_structure

    # -------------------------
    # Low-level analytic helpers
    # -------------------------
    def _assess_competitive_structure(self, models: np.ndarray) -> str:
        """Assess competitive landscape"""
        num_models = len(models)

        if num_models == 2:
            return "COMPETITIVE_STRUCTURE: HEAD_TO_HEAD - Direct two-way comparison"
        elif num_models <= 4:
            return f"COMPETITIVE_STRUCTURE: OLIGOPOLY - {num_models} models competing"
        else:
            return f"COMPETITIVE_STRUCTURE: CROWDED_FIELD - {num_models} models, complex decision space"

    def _detect_correlation_patterns(self, numeric_cols: List[str]) -> List[str]:
        """Detect correlation patterns qualitatively"""
        findings: List[str] = []

        try:
            corr_matrix = self.df[numeric_cols].corr()
        except Exception:
            return findings

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                try:
                    corr = corr_matrix.iloc[i, j]
                    if pd.isna(corr):
                        continue
                    if abs(corr) > 0.7:
                        relationship = "POSITIVE" if corr > 0 else "NEGATIVE"
                        strength = "VERY_STRONG" if abs(corr) > 0.9 else "STRONG"
                        findings.append(
                            f"CORRELATION: {numeric_cols[i]} ↔ {numeric_cols[j]} - "
                            f"{strength}_{relationship} ({abs(corr):.0%} aligned)"
                        )
                except Exception:
                    continue

        return findings

    def _analyze_performance_gaps(self) -> List[str]:
        """Analyze gaps between models"""
        findings: List[str] = []

        metric_cols = [col for col in self.df.columns if any(m in col.lower() for m in ['rmse', 'mae', 'r2', 'auc', 'accuracy'])]

        for metric_col in metric_cols:
            if metric_col in self.df.columns and 'model_name' in self.df.columns:
                try:
                    grouped = self.df.groupby('model_name')[metric_col].mean()
                    if len(grouped) >= 2:
                        # decide direction: lower is better for error metrics
                        lower_better = any(lb in metric_col.lower() for lb in ['rmse', 'mae', 'mse', 'mape'])
                        sorted_models = grouped.sort_values(ascending=lower_better)
                        best_val = float(sorted_models.iloc[0])
                        worst_val = float(sorted_models.iloc[-1])
                        if worst_val == 0:
                            gap = float('inf') if best_val != worst_val else 0.0
                        else:
                            gap = abs((worst_val - best_val) / abs(worst_val))
                        if gap == float('inf'):
                            findings.append(f"PERFORMANCE_GAP_{metric_col.upper()}: COMPUTATION_LIMIT")
                        elif gap > 0.30:
                            findings.append(
                                f"PERFORMANCE_GAP_{metric_col.upper()}: WIDE - Substantial spread between best and worst performers"
                            )
                        elif gap > 0.10:
                            findings.append(
                                f"PERFORMANCE_GAP_{metric_col.upper()}: MODERATE - Meaningful differences between models"
                            )
                        else:
                            findings.append(
                                f"PERFORMANCE_GAP_{metric_col.upper()}: NARROW - Models perform similarly"
                            )
                except Exception:
                    continue

        return findings

    # -------------------------
    # Missing function implementations (deterministic)
    # -------------------------
    def _analyze_cross_metric_consistency(self) -> str:
        """
        Evaluate whether model rankings remain stable across different metrics.
        Returns a qualitative assessment string used by the Comparative layer.
        """

        # No data
        if self.df.empty or "model_name" not in self.df.columns:
            return "CONSISTENCY_ANALYSIS: NO_DATA"

        # Identify potential metric columns
        metric_cols = [
            col for col in self.df.columns
            if any(m in col.lower() for m in ["rmse", "mae", "mse", "mape", "r2", "auc", "accuracy", "precision", "recall", "f1"])
        ]

        if not metric_cols:
            return "CONSISTENCY_ANALYSIS: NO_METRIC_COLUMNS"

        rank_tables: List[Dict[str, float]] = []

        for metric in metric_cols:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.df[metric]):
                continue

            try:
                grouped = self.df.groupby("model_name")[metric].mean()

                # Skip metrics with no variation
                if grouped.nunique() <= 1:
                    continue

                lower_is_better = any(lb in metric.lower() for lb in ["rmse", "mae", "mse", "mape"])

                ranks = grouped.rank(ascending=lower_is_better).to_dict()
                rank_tables.append(ranks)

            except Exception:
                continue

        if len(rank_tables) < 2:
            return "CONSISTENCY_ANALYSIS: NOT_ENOUGH_COMPARABLE_METRICS"

        models = self.df["model_name"].unique()
        variances: Dict[str, float] = {}

        for model in models:
            model_ranks = [rt[model] for rt in rank_tables if model in rt]
            if len(model_ranks) >= 2:
                variances[model] = float(np.var(model_ranks))

        if not variances:
            return "CONSISTENCY_ANALYSIS: NO_VALID_RANK_VARIATION"

        avg_var = float(np.mean(list(variances.values())))

        if avg_var < 0.5:
            return "CONSISTENCY_ANALYSIS: HIGH - Models show similar behavior across metrics"
        elif avg_var < 1.5:
            return "CONSISTENCY_ANALYSIS: MODERATE - Some metrics disagree but overall pattern is stable"
        else:
            return "CONSISTENCY_ANALYSIS: LOW - Rankings change significantly across metrics"

    def _detect_trends(self, time_col: str) -> List[str]:
        """
        Detect simple trends over time for numeric series aggregated by time_col.
        Returns qualitative trend statements; avoids emitting raw numbers.
        """
        findings: List[str] = []

        if self.df.empty or time_col not in self.df.columns:
            return findings

        # Ensure time column is datetime
        try:
            times = pd.to_datetime(self.df[time_col], errors='coerce')
        except Exception:
            return findings

        # pick numeric columns (excluding index-like IDs)
        num_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in ('count',)]
        if not num_cols:
            return findings

        # Create a monthly aggregated view (safe summary)
        try:
            tmp = self.df.copy()
            tmp['_ts'] = pd.to_datetime(tmp[time_col], errors='coerce')
            tmp = tmp.dropna(subset=['_ts'])
            if tmp.empty:
                return findings
            tmp['_month'] = tmp['_ts'].dt.to_period('M').dt.to_timestamp()
            for col in num_cols:
                series = tmp.groupby('_month')[col].mean().dropna()
                if len(series) < 3:
                    continue
                # simple linear trend slope
                x = np.arange(len(series))
                y = series.values
                # robust slope using np.polyfit
                slope = np.polyfit(x, y, 1)[0]
                if abs(slope) < 1e-6:
                    continue
                direction = "UP" if slope > 0 else "DOWN"
                strength = "STRONG" if abs(slope) / (abs(y.mean()) + 1e-9) > 0.05 else "WEAK"
                findings.append(f"TREND_{col.upper()}: {direction}_{strength} - Sustained {direction.lower()} trend detected")
        except Exception:
            pass

        return findings

    def _detect_seasonality(self, time_col: str) -> str:
        """
        Detect simple seasonality by checking month-over-month variation.
        Returns a single qualitative statement or empty string.
        """
        if self.df.empty or time_col not in self.df.columns:
            return ""

        try:
            tmp = self.df.copy()
            tmp['_ts'] = pd.to_datetime(tmp[time_col], errors='coerce')
            tmp = tmp.dropna(subset=['_ts'])
            if tmp.empty:
                return ""

            num_cols = [c for c in tmp.select_dtypes(include=[np.number]).columns]
            if not num_cols:
                return ""

            # compute monthly coefficient of variation across months for top numeric column
            col = num_cols[0]
            monthly = tmp.groupby(tmp['_ts'].dt.month)[col].mean()
            if monthly.empty or len(monthly) < 6:
                return ""
            cv = monthly.std() / (monthly.mean() + 1e-9)
            if cv > 0.15:
                return f"SEASONALITY_{col.upper()}: EVIDENT - Monthly pattern present (cv={cv:.2f})"
            elif cv > 0.07:
                return f"SEASONALITY_{col.upper()}: MODEST - Some seasonal variation"
            else:
                return ""
        except Exception:
            return ""

    def _assess_momentum(self, time_col: str) -> str:
        """
        Assess recent momentum: is a recent window accelerating or decelerating.
        Returns a qualitative statement.
        """
        if self.df.empty or time_col not in self.df.columns:
            return ""

        try:
            tmp = self.df.copy()
            tmp['_ts'] = pd.to_datetime(tmp[time_col], errors='coerce')
            tmp = tmp.dropna(subset=['_ts'])
            if tmp.empty:
                return ""

            num_cols = [c for c in tmp.select_dtypes(include=[np.number]).columns]
            if not num_cols:
                return ""

            col = num_cols[0]
            series = tmp.sort_values('_ts').groupby('_ts')[col].mean()
            if len(series) < 6:
                return ""

            recent = series[-3:].mean()
            prior = series[-6:-3].mean()
            if prior == 0:
                return ""

            pct_change = (recent - prior) / (abs(prior) + 1e-9)
            if pct_change > 0.05:
                return f"MOMENTUM_{col.upper()}: ACCELERATING - Recent period shows >5% increase"
            elif pct_change < -0.05:
                return f"MOMENTUM_{col.upper()}: DECELERATING - Recent period shows >5% decrease"
            else:
                return f"MOMENTUM_{col.upper()}: STABLE - No strong short-term momentum"
        except Exception:
            return ""

    def _detect_statistical_outliers(self) -> List[str]:
        """Detect outliers using IQR and z-score heuristics; return qualitative notes"""
        findings: List[str] = []
        if self.df.empty:
            return findings

        numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns]
        for col in numeric_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = ((series < lower) | (series > upper)).sum()
            pct = outlier_count / (len(series) + 1e-9)
            if pct > 0.05:
                findings.append(f"OUTLIERS_{col.upper()}: MODERATE - {outlier_count} values (~{pct:.1%}) flagged by IQR")
            elif pct > 0.01:
                findings.append(f"OUTLIERS_{col.upper()}: LOW - small number of outliers detected")
        return findings

    def _detect_unexpected_patterns(self) -> List[str]:
        """Detect unexpected patterns such as sudden reversals across metrics"""
        findings: List[str] = []
        if self.df.empty:
            return findings

        # Example heuristic: if a model that usually ranks high suddenly ranks low in recent executions
        if 'model_name' in self.df.columns and any(col for col in self.df.columns if 'metric' in col.lower() or True):
            try:
                # If there's a 'prediction_date' or timestamp column, use it for recency
                date_cols = [c for c in self.df.columns if 'date' in c.lower() or 'time' in c.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    tmp = self.df.dropna(subset=[date_col])
                    if not tmp.empty and 'model_name' in tmp.columns:
                        recent_period = tmp.sort_values(date_col).tail(100)
                        if len(recent_period) >= 10:
                            # quick check: top model historically vs recently
                            metric_cols = [c for c in tmp.select_dtypes(include=[np.number]).columns if c not in (date_col,)]
                            if metric_cols:
                                col = metric_cols[0]
                                hist_best = self.df.groupby('model_name')[col].mean().idxmax()
                                recent_best = recent_period.groupby('model_name')[col].mean().idxmax()
                                if hist_best != recent_best:
                                    findings.append(f"UNEXPECTED_PATTERN: Performance leader changed from {hist_best} to {recent_best} recently")
                # fallback: check for sudden spikes in numeric cols
                numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns]
                for col in numeric_cols[:3]:
                    series = self.df[col].dropna()
                    if len(series) < 10:
                        continue
                    if series.tail(5).mean() > series.mean() * 1.5:
                        findings.append(f"UNEXPECTED_PATTERN_{col.upper()}: RECENT_SPIKE - last values notably higher than historic average")
            except Exception:
                pass
        return findings

    def _assess_data_quality(self) -> List[str]:
        """Check missingness, duplicates, and anomalous distributions"""
        findings: List[str] = []
        if self.df.empty:
            return findings

        # Missingness
        missing = self.df.isna().mean()
        high_missing = missing[missing > 0.2].index.tolist()
        if high_missing:
            findings.append(f"DATA_QUALITY: HIGH_MISSING - Columns with >20% missing: {', '.join(high_missing)}")

        # Duplicates
        try:
            dup_count = self.df.duplicated().sum()
            if dup_count > 0:
                findings.append(f"DATA_QUALITY: DUPLICATES - {dup_count} exact duplicate rows detected")
        except Exception:
            pass

        # Unexpected constant columns
        const_cols = [c for c in self.df.columns if self.df[c].nunique(dropna=True) <= 1]
        if const_cols:
            findings.append(f"DATA_QUALITY: CONSTANT_COLUMNS - {', '.join(const_cols)} have single unique value")

        return findings

    # -------------------------
    # Business-context interpreters
    # -------------------------
    def _interpret_forecasting_context(self) -> List[str]:
        """Interpretation tailored for forecasting (NRx) use cases"""
        findings: List[str] = []
        # Look for forecasting-specific signals: time series columns, TRx/NRx predictions
        time_cols = [c for c in self.df.columns if 'date' in c.lower() or 'time' in c.lower()]
        pred_cols = [c for c in self.df.columns if any(k in c.lower() for k in ['trx', 'nrx', 'prediction', 'forecast'])]
        if time_cols and pred_cols:
            findings.append("FORECASTING_CONTEXT: Time-series forecasting data present; check horizon and confidence intervals before deployment")
        else:
            findings.append("FORECASTING_CONTEXT: Forecasting signals incomplete; confirm prediction dates and horizon")
        return findings

    def _interpret_hcp_context(self) -> List[str]:
        """Interpretation for HCP engagement use cases"""
        findings: List[str] = []
        if 'HCP' in "".join(map(str, self.df.columns)).upper() or any('hcp' in c.lower() for c in self.df.columns):
            findings.append("HCP_CONTEXT: HCP-level features present; segment-level targeting feasible")
        else:
            findings.append("HCP_CONTEXT: HCP identifiers/features missing; caution when recommending targeting strategies")
        return findings

    def _interpret_drift_context(self) -> List[str]:
        """Interpretation for drift detection use cases"""
        findings: List[str] = []
        # Look for drift indicators
        drift_cols = [c for c in self.df.columns if 'drift' in c.lower() or 'psi' in c.lower() or 'ks' in c.lower()]
        if drift_cols:
            findings.append("DRIFT_CONTEXT: Drift metrics present; schedule retraining if trends persist")
        else:
            findings.append("DRIFT_CONTEXT: No explicit drift metrics found; consider computing PSI/K-S vs baseline")
        return findings

    def _interpret_ensemble_context(self) -> List[str]:
        """Interpretation for ensemble comparisons"""
        findings: List[str] = []
        if 'ensemble' in "".join(map(str, self.df.columns)).lower() or any('ensemble' in str(x).lower() for x in self.df.get('model_name', []) if isinstance(x, str)):
            findings.append("ENSEMBLE_CONTEXT: Ensemble models present; validate diversity of base learners before deployment")
        else:
            findings.append("ENSEMBLE_CONTEXT: No ensembles detected; ensemble-specific guidance not applicable")
        return findings

    # -------------------------
    # Decision framework helpers
    # -------------------------
    def _assess_decision_clarity(self) -> Dict[str, Any]:
        """
        Returns a simple dict with clarity score (0-1) and description.
        Score based on: cross-metric consistency + number of critical findings
        """
        score = 0.0
        reasons: List[str] = []

        # Start with consistency
        consistency_str = self._analyze_cross_metric_consistency()
        if "HIGH" in consistency_str:
            score += 0.6
            reasons.append("High cross-metric consistency")
        elif "MODERATE" in consistency_str:
            score += 0.35
            reasons.append("Moderate cross-metric consistency")
        elif "LOW" in consistency_str:
            score += 0.1
            reasons.append("Low cross-metric consistency")

        # Penalize for critical anomalies
        anomaly_layer = self._detect_anomalies()
        critical_count = sum(1 for f in anomaly_layer.findings if 'SEVERE' in f or 'RECENT_SPIKE' in f)
        score -= 0.1 * min(critical_count, 3)

        score = max(0.0, min(1.0, score))
        description = f"DECISION_CLARITY_SCORE: {score:.2f} - {'; '.join(reasons) if reasons else 'No strong signals'}"
        return {"score": score, "description": description}

    def _identify_decision_blockers(self) -> List[str]:
        """Identify blockers that would prevent immediate deployment"""
        blockers: List[str] = []
        dq = self._assess_data_quality()
        if any('HIGH_MISSING' in s for s in dq):
            blockers.append("BLOCKER: High missingness in key columns")
        anomalies = self._detect_statistical_outliers()
        if anomalies:
            blockers.append("BLOCKER: Outliers detected; investigate root causes before deployment")
        consistency = self._analyze_cross_metric_consistency()
        if "LOW" in consistency:
            blockers.append("BLOCKER: Conflicting metric rankings; requires prioritization of business metric")
        return blockers

    def _build_recommendation_framework(self) -> List[str]:
        """Produce prioritized actions: quick wins, medium, long-term"""
        recs: List[str] = []

        # Quick wins
        recs.append("RECOMMENDATION_QUICK: Run sanity checks and data quality fixes (missing values, duplicates)")
        recs.append("RECOMMENDATION_QUICK: Validate top-performing model on holdout or recent production slice")

        # Mid-term
        consistency = self._analyze_cross_metric_consistency()
        if "HIGH" in consistency:
            recs.append("RECOMMENDATION_MEDIUM: Consider deployment candidate (monitor for drift)")
        elif "MODERATE" in consistency:
            recs.append("RECOMMENDATION_MEDIUM: Run targeted A/B tests comparing top models")
        else:
            recs.append("RECOMMENDATION_MEDIUM: Resolve metric conflicts; align on business objective")

        # Long-term
        recs.append("RECOMMENDATION_LONG: Establish monitoring pipeline (drift, performance, data quality)")
        recs.append("RECOMMENDATION_LONG: Document features and feature engineering for reproducibility")

        return recs

    def _assess_deployment_risks(self) -> List[str]:
        """Return a short risk register with qualitative severity"""
        risks: List[str] = []
        dq = self._assess_data_quality()
        if dq:
            risks.append("RISK_DATA: Data quality issues may lead to model degradation (MEDIUM)")
        consistency = self._analyze_cross_metric_consistency()
        if "LOW" in consistency:
            risks.append("RISK_METRICS: Metric disagreement increases operational risk (HIGH)")
        # add drift risk if drift metrics exist
        drift_cols = [c for c in self.df.columns if 'drift' in c.lower() or 'drift_score' in c.lower()]
        if drift_cols:
            risks.append("RISK_DRIFT: Evidence of drift requires retraining cadence (HIGH)")
        if not risks:
            risks.append("RISK_NONE_IDENTIFIED: No immediate, high-severity risks detected (LOW)")
        return risks

    # -------------------------
    # Synthesis & presentation helpers
    # -------------------------
    def _generate_executive_summary(self, layers: List[AnalyticalLayer]) -> str:
        """Create a 2-4 sentence executive summary synthesizing the most critical points"""
        if not layers:
            return "EXECUTIVE_SUMMARY: No analysis available."

        # Use highest-priority layer findings
        top_layers = [l for l in layers if l.business_relevance in ("CRITICAL", "HIGH")]
        top_findings = []
        for l in top_layers[:3]:
            if l.findings:
                top_findings.append(l.findings[0])

        # fallback to first layer findings
        if not top_findings and layers and layers[0].findings:
            top_findings.append(layers[0].findings[0])

        sentences = []
        sentences.append(f"Based on deterministic analysis across {len(layers)} analytic layers, key signals were identified.")
        for f in top_findings[:2]:
            # keep sentences concise
            sentences.append(f"{f}.")

        # close with high-level recommendation
        clarity = self._assess_decision_clarity()
        if clarity.get('score', 0) > 0.6:
            sentences.append("Recommendation: Proceed to controlled deployment with monitoring.")
        else:
            sentences.append("Recommendation: Resolve noted data or metric conflicts before wide deployment.")

        return " ".join(sentences)

    def _extract_key_findings(self, layers: List[AnalyticalLayer]) -> List[str]:
        """Return a prioritized list of key findings suitable for bulleting"""
        findings: List[str] = []
        # prioritize critical layers
        for layer in sorted(layers, key=lambda x: (x.business_relevance != "CRITICAL", x.confidence != "HIGH")):
            for f in layer.findings:
                findings.append(f"{layer.layer_name}: {f}")
                if len(findings) >= 12:
                    return findings
        return findings

    def _format_detailed_analysis(self, layers: List[AnalyticalLayer]) -> Dict[str, Any]:
        """Return a structured mapping of layer -> findings and metadata"""
        detail: Dict[str, Any] = {}
        for layer in layers:
            detail[layer.layer_name] = {
                "findings": layer.findings,
                "confidence": layer.confidence,
                "business_relevance": layer.business_relevance
            }
        return detail

    def _consolidate_recommendations(self, layers: List[AnalyticalLayer]) -> List[str]:
        """Collect and prioritize recommendations across layers"""
        # prefer recommendations from decision layer if present
        recs: List[str] = []
        for layer in layers:
            if layer.layer_name == "DECISION_FRAMEWORK":
                recs.extend(layer.findings)
        # add other pragmatic items if space
        if len(recs) < 6:
            # add quick data quality items
            dq = self._assess_data_quality()
            recs.extend([f"DATA_ACTION: {x}" for x in dq][:3])
        # dedupe and limit
        seen = set()
        out = []
        for r in recs:
            if r not in seen:
                seen.add(r)
                out.append(r)
            if len(out) >= 8:
                break
        return out

    def _assess_overall_confidence(self, layers: List[AnalyticalLayer]) -> Dict[str, Any]:
        """
        Produce a confidence summary using counts of HIGH confidences and a simple score.
        Score in [0,1].
        """
        if not layers:
            return {"score": 0.0, "summary": "No layers evaluated."}

        high_count = sum(1 for l in layers if l.confidence == "HIGH")
        total = len(layers)
        score = (high_count / total) if total > 0 else 0.0

        if score > 0.66:
            level = "HIGH"
        elif score > 0.33:
            level = "MEDIUM"
        else:
            level = "LOW"

        summary = f"Overall confidence: {level} (score={score:.2f}) based on {high_count}/{total} high-confidence layers."

        return {"score": float(score), "summary": summary}