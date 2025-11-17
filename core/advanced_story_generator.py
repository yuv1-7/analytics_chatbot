from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

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
        self.context = query_context
        self.layers = []
    
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
    
    def _analyze_statistical_patterns(self) -> AnalyticalLayer:
        """Layer 1: Pure statistical patterns"""
        findings = []
        
        if self.df.empty:
            return AnalyticalLayer("STATISTICAL", ["NO_DATA"], "LOW", "LOW")
        
        # Distribution analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            skewness = self.df[col].skew()
            kurtosis = self.df[col].kurtosis()
            
            if abs(skewness) > 1:
                direction = "RIGHT" if skewness > 0 else "LEFT"
                findings.append(
                    f"DISTRIBUTION_{col.upper()}: {direction}_SKEWED - "
                    f"Indicates {'a few high performers' if skewness > 0 else 'a few underperformers'}"
                )
            
            if kurtosis > 3:
                findings.append(
                    f"DISTRIBUTION_{col.upper()}: HEAVY_TAILED - "
                    f"Outliers present, potential for extreme values"
                )
        
        # Correlation patterns (qualitative only)
        if len(numeric_cols) >= 2:
            correlations = self._detect_correlation_patterns(numeric_cols)
            findings.extend(correlations)
        
        return AnalyticalLayer(
            layer_name="STATISTICAL_PATTERNS",
            findings=findings,
            confidence="HIGH",
            business_relevance="MODERATE"
        )
    
    def _analyze_comparative_patterns(self) -> AnalyticalLayer:
        """Layer 2: Model-to-model comparisons"""
        findings = []
        
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
        
        # Consistency analysis
        consistency = self._analyze_cross_metric_consistency()
        findings.append(consistency)
        
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
        
        # Trend detection
        trends = self._detect_trends(time_cols[0])
        findings.extend(trends)
        
        # Seasonality
        seasonality = self._detect_seasonality(time_cols[0])
        if seasonality:
            findings.append(seasonality)
        
        # Momentum
        momentum = self._assess_momentum(time_cols[0])
        findings.append(momentum)
        
        return AnalyticalLayer(
            layer_name="TEMPORAL_TRENDS",
            findings=findings,
            confidence="MEDIUM",
            business_relevance="HIGH"
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
        findings = []
        
        use_case = self.context.get('use_case', '').lower()
        comparison_type = self.context.get('comparison_type', '').lower()
        
        # Use case specific interpretations
        if 'nrx' in use_case or 'forecast' in use_case:
            findings.extend(self._interpret_forecasting_context())
        elif 'hcp' in use_case or 'engagement' in use_case:
            findings.extend(self._interpret_hcp_context())
        elif 'drift' in comparison_type:
            findings.extend(self._interpret_drift_context())
        elif 'ensemble' in comparison_type:
            findings.extend(self._interpret_ensemble_context())
        
        return AnalyticalLayer(
            layer_name="BUSINESS_CONTEXT",
            findings=findings,
            confidence="MEDIUM",
            business_relevance="CRITICAL"
        )
    
    def _generate_decision_framework(self) -> AnalyticalLayer:
        """Layer 6: Decision support"""
        findings = []
        
        # Assess decision clarity
        clarity = self._assess_decision_clarity()
        findings.append(clarity['description'])
        
        # Identify decision blockers
        blockers = self._identify_decision_blockers()
        if blockers:
            findings.extend(blockers)
        
        # Generate recommendation framework
        framework = self._build_recommendation_framework()
        findings.extend(framework)
        
        # Risk assessment
        risks = self._assess_deployment_risks()
        if risks:
            findings.extend(risks)
        
        return AnalyticalLayer(
            layer_name="DECISION_FRAMEWORK",
            findings=findings,
            confidence="HIGH",
            business_relevance="CRITICAL"
        )
    
    def _synthesize_layers(self, layers: List[AnalyticalLayer]) -> Dict[str, Any]:
        """Synthesize all layers into coherent story structure"""
        
        # Priority ranking
        priority_order = {
            "CRITICAL": 1,
            "HIGH": 2,
            "MODERATE": 3,
            "LOW": 4
        }
        
        # Sort layers by business relevance
        sorted_layers = sorted(
            layers,
            key=lambda x: priority_order.get(x.business_relevance, 5)
        )
        
        # Build narrative structure
        narrative_structure = {
            "executive_summary": self._generate_executive_summary(sorted_layers),
            "key_findings": self._extract_key_findings(sorted_layers),
            "detailed_analysis": self._format_detailed_analysis(sorted_layers),
            "recommendations": self._consolidate_recommendations(sorted_layers),
            "confidence_assessment": self._assess_overall_confidence(sorted_layers),
            "metadata": {
                "layers_analyzed": len(layers),
                "high_confidence_findings": sum(1 for l in layers if l.confidence == "HIGH"),
                "critical_findings": sum(1 for l in layers if l.business_relevance == "CRITICAL")
            }
        }
        
        return narrative_structure
    
    # Helper methods
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
        findings = []
        
        # Calculate correlations
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strong correlations
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) > 0.7:
                    relationship = "POSITIVE" if corr > 0 else "NEGATIVE"
                    strength = "VERY_STRONG" if abs(corr) > 0.9 else "STRONG"
                    
                    findings.append(
                        f"CORRELATION: {numeric_cols[i]} ↔ {numeric_cols[j]} - "
                        f"{strength}_{relationship} ({abs(corr):.0%} aligned)"
                    )
        
        return findings
    
    def _analyze_performance_gaps(self) -> List[str]:
        """Analyze gaps between models"""
        findings = []
        
        metric_cols = [col for col in self.df.columns if any(m in col.lower() for m in ['rmse', 'mae', 'r2', 'auc'])]
        
        for metric_col in metric_cols:
            if metric_col in self.df.columns and 'model_name' in self.df.columns:
                grouped = self.df.groupby('model_name')[metric_col].mean()
                
                if len(grouped) >= 2:
                    sorted_models = grouped.sort_values()
                    best = sorted_models.iloc[0]
                    worst = sorted_models.iloc[-1]
                    gap = abs((worst - best) / worst)
                    
                    if gap > 0.20:
                        findings.append(
                            f"PERFORMANCE_GAP_{metric_col.upper()}: WIDE - "
                            f"Substantial spread between best and worst performers"
                        )
                    elif gap > 0.10:
                        findings.append(
                            f"PERFORMANCE_GAP_{metric_col.upper()}: MODERATE - "
                            f"Meaningful differences between models"
                        )
                    else:
                        findings.append(
                            f"PERFORMANCE_GAP_{metric_col.upper()}: NARROW - "
                            f"Models perform similarly"
                        )
        
        return findings