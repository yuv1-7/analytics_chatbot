"""
Fills placeholders in insight templates with actual computed values
NO LLM involved - pure string replacement
"""

import re
from typing import Dict, Any, List
import numpy as np


class PlaceholderFiller:
    """Fills template placeholders with actual values"""
    
    def __init__(self, template: str, analysis_results: Dict[str, Any]):
        """
        Initialize filler
        
        Args:
            template: Template string with {{PLACEHOLDERS}}
            analysis_results: Analysis results with computed_values
        """
        self.template = template
        self.analysis_results = analysis_results
        self.computed_values = analysis_results.get('computed_values', {})
    
    def fill(self) -> str:
        """
        Fill all placeholders in template
        
        Returns:
            Filled template string
        """
        filled = self.template
        
        # Extract analysis type
        analysis_type = self.analysis_results.get('analysis_type', 'general')
        
        # Route to appropriate filler
        if analysis_type == 'performance_comparison':
            filled = self._fill_performance_comparison(filled)
        elif analysis_type == 'drift_analysis':
            filled = self._fill_drift_analysis(filled)
        elif analysis_type == 'ensemble_vs_base':
            filled = self._fill_ensemble_vs_base(filled)
        elif analysis_type == 'feature_importance':
            filled = self._fill_feature_importance(filled)
        elif analysis_type == 'uplift_analysis':
            filled = self._fill_uplift_analysis(filled)
        else:
            filled = self._fill_general(filled)
        
        # Clean up any unfilled placeholders
        filled = self._cleanup_unfilled(filled)
        
        return filled
    
    # ========================================================================
    # PERFORMANCE COMPARISON FILLERS
    # ========================================================================
    
    def _fill_performance_comparison(self, template: str) -> str:
        """Fill performance comparison placeholders"""
        filled = template
        
        models = self.analysis_results.get('models_analyzed', [])
        metrics_summary = self.computed_values.get('metrics', {})
        rankings = self.analysis_results.get('rankings', {})
        
        # Basic info
        filled = filled.replace('{{NUM_MODELS}}', str(len(models)))
        filled = filled.replace('{{MODEL_NAMES}}', ', '.join(models))
        
        # Best model identification
        if rankings:
            first_metric = list(rankings.keys())[0]
            first_ranking = rankings[first_metric]
            if first_ranking:
                best_model, best_value = first_ranking[0]
                filled = filled.replace('{{BEST_MODEL}}', best_model)
                filled = filled.replace('{{BEST_METRIC_NAME}}', first_metric.upper())
                filled = filled.replace('{{BEST_METRIC_VALUE}}', f"{best_value:.3f}")
        
        # Metric rankings table
        rankings_table = self._build_rankings_table(rankings)
        filled = filled.replace('{{METRIC_RANKINGS_TABLE}}', rankings_table)
        
        # Performance summary table
        summary_table = self._build_performance_summary_table(metrics_summary)
        filled = filled.replace('{{PERFORMANCE_SUMMARY_TABLE}}', summary_table)
        
        # Recommendations
        recommendations = self._generate_performance_recommendations(rankings, metrics_summary)
        filled = filled.replace('{{RECOMMENDATIONS}}', recommendations)
        
        return filled
    
    def _build_rankings_table(self, rankings: Dict[str, List]) -> str:
        """Build markdown table for rankings"""
        if not rankings:
            return "*No ranking data available*"
        
        lines = []
        
        for metric, ranked_models in rankings.items():
            lines.append(f"### {metric.upper()}\n")
            lines.append("| Rank | Model | Value |\n")
            lines.append("|------|-------|-------|\n")
            
            for rank, (model, value) in enumerate(ranked_models, 1):
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
                lines.append(f"| {rank} {emoji} | {model} | {value:.4f} |\n")
            
            lines.append("\n")
        
        return "".join(lines)
    
    def _build_performance_summary_table(self, metrics_summary: Dict) -> str:
        """Build comprehensive performance summary table"""
        if not metrics_summary:
            return "*No performance data available*"
        
        lines = []
        lines.append("| Model | Metric | Mean | Std Dev | Min | Max |\n")
        lines.append("|-------|--------|------|---------|-----|-----|\n")
        
        for metric, model_values in metrics_summary.items():
            for model, stats in model_values.items():
                lines.append(f"| {model} | {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
        
        return "".join(lines)
    
    def _generate_performance_recommendations(self, rankings: Dict, metrics_summary: Dict) -> str:
        """Generate recommendations based on performance"""
        recommendations = []
        
        if not rankings:
            return "*Insufficient data for recommendations*"
        
        # Find overall best model (appears most in top positions)
        model_scores = {}
        for metric, ranked_models in rankings.items():
            for rank, (model, value) in enumerate(ranked_models, 1):
                if model not in model_scores:
                    model_scores[model] = 0
                # Give points inversely proportional to rank
                model_scores[model] += (len(ranked_models) - rank + 1)
        
        if model_scores:
            best_overall = max(model_scores, key=model_scores.get)
            recommendations.append(f"1. **Deploy {best_overall}** - Shows strongest overall performance across metrics")
        
        # Check for consistency
        if metrics_summary:
            for metric, model_values in metrics_summary.items():
                for model, stats in model_values.items():
                    if stats['std'] > stats['mean'] * 0.3:  # High variance
                        recommendations.append(f"2. **Monitor {model}** - Shows high variability in {metric} (std={stats['std']:.3f})")
                        break
        
        # General recommendation
        recommendations.append("3. **Continue monitoring** - Track performance trends over time to detect degradation")
        
        return "\n".join(recommendations)
    
    # ========================================================================
    # DRIFT ANALYSIS FILLERS
    # ========================================================================
    
    def _fill_drift_analysis(self, template: str) -> str:
        """Fill drift analysis placeholders"""
        filled = template
        
        drift_summary = self.computed_values.get('drift_summary', {})
        models_with_drift = self.computed_values.get('models_with_drift', [])
        
        # Basic stats
        total_models = len(drift_summary)
        num_with_drift = len(models_with_drift)
        drift_pct = (num_with_drift / total_models * 100) if total_models > 0 else 0
        
        filled = filled.replace('{{NUM_MODELS_WITH_DRIFT}}', str(num_with_drift))
        filled = filled.replace('{{TOTAL_MODELS}}', str(total_models))
        filled = filled.replace('{{DRIFT_PERCENTAGE}}', f"{drift_pct:.1f}")
        
        # Drift models table
        drift_table = self._build_drift_table(drift_summary, models_with_drift)
        filled = filled.replace('{{DRIFT_MODELS_TABLE}}', drift_table)
        
        # Drift scores chart (text representation)
        drift_chart = self._build_drift_chart(drift_summary)
        filled = filled.replace('{{DRIFT_SCORES_CHART}}', drift_chart)
        
        # Recommendations
        recommendations = self._generate_drift_recommendations(drift_summary, models_with_drift)
        filled = filled.replace('{{DRIFT_RECOMMENDATIONS}}', recommendations)
        
        return filled
    
    def _build_drift_table(self, drift_summary: Dict, models_with_drift: List[str]) -> str:
        """Build table of models with drift"""
        if not models_with_drift:
            return "*No models with detected drift* ✅"
        
        lines = []
        lines.append("| Model | Mean Drift Score | Max Drift Score | Status |\n")
        lines.append("|-------|------------------|-----------------|--------|\n")
        
        for model in models_with_drift:
            if model in drift_summary:
                stats = drift_summary[model]
                status = "⚠️ Drift Detected" if stats['drift_detected'] else "✅ OK"
                lines.append(f"| {model} | {stats['mean_drift_score']:.4f} | {stats['max_drift_score']:.4f} | {status} |\n")
        
        return "".join(lines)
    
    def _build_drift_chart(self, drift_summary: Dict) -> str:
        """Build text-based drift score chart"""
        if not drift_summary:
            return "*No drift data available*"
        
        lines = []
        lines.append("```\n")
        lines.append("Drift Score Distribution:\n")
        lines.append("-" * 50 + "\n")
        
        for model, stats in sorted(drift_summary.items(), key=lambda x: x[1]['mean_drift_score'], reverse=True):
            score = stats['mean_drift_score']
            bar_length = int(score * 50)  # Scale to 50 chars max
            bar = "█" * bar_length
            lines.append(f"{model:20s} | {bar} {score:.3f}\n")
        
        lines.append("```\n")
        
        return "".join(lines)
    
    def _generate_drift_recommendations(self, drift_summary: Dict, models_with_drift: List[str]) -> str:
        """Generate drift-specific recommendations"""
        recommendations = []
        
        if not models_with_drift:
            recommendations.append("✅ **No immediate action required** - All models are stable")
            return "\n".join(recommendations)
        
        # Priority 1: High drift models
        high_drift = [m for m, s in drift_summary.items() if s['mean_drift_score'] > 0.15]
        if high_drift:
            recommendations.append(f"1. **Immediate Retraining Required** for: {', '.join(high_drift)}")
        
        # Priority 2: Medium drift models
        medium_drift = [m for m, s in drift_summary.items() if 0.10 < s['mean_drift_score'] <= 0.15]
        if medium_drift:
            recommendations.append(f"2. **Schedule Retraining** for: {', '.join(medium_drift)}")
        
        # Priority 3: Monitor
        low_drift = [m for m, s in drift_summary.items() if s['mean_drift_score'] <= 0.10 and s['drift_detected']]
        if low_drift:
            recommendations.append(f"3. **Continue Monitoring** for: {', '.join(low_drift)}")
        
        # General
        recommendations.append("4. **Root Cause Analysis** - Investigate data distribution changes")
        
        return "\n".join(recommendations)
    
    # ========================================================================
    # ENSEMBLE VS BASE FILLERS
    # ========================================================================
    
    def _fill_ensemble_vs_base(self, template: str) -> str:
        """Fill ensemble vs base comparison placeholders"""
        filled = template
        
        ensemble_metrics = self.computed_values.get('ensemble_metrics', {})
        base_metrics = self.computed_values.get('base_metrics', {})
        ensemble_advantage = self.computed_values.get('ensemble_advantage', {})
        
        # Comparison table
        comparison_table = self._build_ensemble_comparison_table(
            ensemble_metrics, base_metrics, ensemble_advantage
        )
        filled = filled.replace('{{ENSEMBLE_VS_BASE_TABLE}}', comparison_table)
        
        # Advantage breakdown
        advantage_breakdown = self._build_advantage_breakdown(ensemble_advantage)
        filled = filled.replace('{{ADVANTAGE_BREAKDOWN}}', advantage_breakdown)
        
        # Analysis
        analysis = self._generate_ensemble_analysis(ensemble_advantage)
        filled = filled.replace('{{ENSEMBLE_ANALYSIS}}', analysis)
        
        # Recommendations
        recommendations = self._generate_ensemble_recommendations(ensemble_advantage)
        filled = filled.replace('{{ENSEMBLE_RECOMMENDATIONS}}', recommendations)
        
        return filled
    
    def _build_ensemble_comparison_table(self, ensemble_metrics: Dict, 
                                          base_metrics: Dict, 
                                          ensemble_advantage: Dict) -> str:
        """Build ensemble vs base comparison table"""
        if not ensemble_metrics or not base_metrics:
            return "*Insufficient data for comparison*"
        
        lines = []
        lines.append("| Metric | Ensemble | Base (Avg) | Difference | Better? |\n")
        lines.append("|--------|----------|------------|------------|----------|\n")
        
        for metric in ensemble_metrics:
            if metric in base_metrics and metric in ensemble_advantage:
                ens_val = ensemble_metrics[metric]
                base_val = base_metrics[metric]
                diff = ensemble_advantage[metric]['absolute_diff']
                pct = ensemble_advantage[metric]['percent_change']
                better = ensemble_advantage[metric]['better']
                
                better_icon = "✅" if better else "❌"
                diff_str = f"{diff:+.4f} ({pct:+.1f}%)"
                
                lines.append(f"| {metric} | {ens_val:.4f} | {base_val:.4f} | {diff_str} | {better_icon} |\n")
        
        return "".join(lines)
    
    def _build_advantage_breakdown(self, ensemble_advantage: Dict) -> str:
        """Build detailed advantage breakdown"""
        if not ensemble_advantage:
            return "*No advantage data available*"
        
        lines = []
        
        better_count = sum(1 for v in ensemble_advantage.values() if v['better'])
        total_count = len(ensemble_advantage)
        
        lines.append(f"**Ensemble wins on {better_count}/{total_count} metrics ({better_count/total_count*100:.0f}%)**\n\n")
        
        # Winners
        winners = [m for m, v in ensemble_advantage.items() if v['better']]
        if winners:
            lines.append("**Metrics where ensemble wins:**\n")
            for metric in winners:
                pct = ensemble_advantage[metric]['percent_change']
                lines.append(f"- {metric}: {abs(pct):.1f}% improvement\n")
            lines.append("\n")
        
        # Losers
        losers = [m for m, v in ensemble_advantage.items() if not v['better']]
        if losers:
            lines.append("**Metrics where base models win:**\n")
            for metric in losers:
                pct = ensemble_advantage[metric]['percent_change']
                lines.append(f"- {metric}: {abs(pct):.1f}% worse\n")
        
        return "".join(lines)
    
    def _generate_ensemble_analysis(self, ensemble_advantage: Dict) -> str:
        """Generate analysis text"""
        if not ensemble_advantage:
            return "*Insufficient data for analysis*"
        
        better_count = sum(1 for v in ensemble_advantage.values() if v['better'])
        total_count = len(ensemble_advantage)
        win_rate = better_count / total_count
        
        lines = []
        
        if win_rate >= 0.8:
            lines.append("The ensemble demonstrates **strong superiority** over base models, winning on the vast majority of metrics. ")
            lines.append("This suggests effective model diversity and meta-learner optimization.\n\n")
        elif win_rate >= 0.5:
            lines.append("The ensemble shows **mixed performance** compared to base models. ")
            lines.append("It excels on some metrics but lags on others, suggesting opportunities for ensemble refinement.\n\n")
        else:
            lines.append("Base models **outperform the ensemble** on most metrics. ")
            lines.append("This may indicate insufficient diversity among base models, overfitting, or suboptimal meta-learner configuration.\n\n")
        
        # Magnitude analysis
        avg_improvement = np.mean([abs(v['percent_change']) for v in ensemble_advantage.values()])
        lines.append(f"Average performance difference: **{avg_improvement:.1f}%**\n")
        
        return "".join(lines)
    
    def _generate_ensemble_recommendations(self, ensemble_advantage: Dict) -> str:
        """Generate ensemble recommendations"""
        recommendations = []
        
        if not ensemble_advantage:
            return "*Insufficient data for recommendations*"
        
        better_count = sum(1 for v in ensemble_advantage.values() if v['better'])
        total_count = len(ensemble_advantage)
        win_rate = better_count / total_count
        
        if win_rate >= 0.8:
            recommendations.append("1. **Deploy Ensemble** - Clear performance advantage justifies complexity")
            recommendations.append("2. **Monitor Individual Models** - Track base model contributions over time")
        elif win_rate >= 0.5:
            recommendations.append("1. **Selective Use** - Deploy ensemble for metrics where it excels")
            recommendations.append("2. **Investigate Underperformance** - Analyze why ensemble lags on certain metrics")
            recommendations.append("3. **Consider Refinement** - Adjust meta-learner or base model selection")
        else:
            recommendations.append("1. **Use Best Base Model** - Ensemble adds complexity without benefit")
            recommendations.append("2. **Root Cause Analysis** - Investigate lack of diversity or overfitting")
            recommendations.append("3. **Redesign Ensemble** - Consider different base models or ensemble approach")
        
        return "\n".join(recommendations)
    
    # ========================================================================
    # FEATURE IMPORTANCE FILLERS
    # ========================================================================
    
    def _fill_feature_importance(self, template: str) -> str:
        """Fill feature importance placeholders"""
        filled = template
        
        top_features = self.computed_values.get('top_features', {})
        
        # Top features table
        features_table = self._build_features_table(top_features)
        filled = filled.replace('{{TOP_FEATURES_TABLE}}', features_table)
        
        # Insights
        insights = self._generate_feature_insights(top_features)
        filled = filled.replace('{{FEATURE_INSIGHTS}}', insights)
        
        # Recommendations
        recommendations = self._generate_feature_recommendations(top_features)
        filled = filled.replace('{{FEATURE_RECOMMENDATIONS}}', recommendations)
        
        return filled
    
    def _build_features_table(self, top_features: Dict) -> str:
        """Build feature importance table"""
        if not top_features:
            return "*No feature importance data available*"
        
        lines = []
        lines.append("| Rank | Feature | Importance Score |\n")
        lines.append("|------|---------|------------------|\n")
        
        # Sort by importance score
        sorted_features = sorted(top_features.items(), 
                                key=lambda x: x[1]['importance_score'], 
                                reverse=True)
        
        for rank, (feature, data) in enumerate(sorted_features, 1):
            score = data['importance_score']
            bar = "█" * int(score * 20)  # Visual bar
            lines.append(f"| {rank} | {feature} | {score:.4f} {bar} |\n")
        
        return "".join(lines)
    
    def _generate_feature_insights(self, top_features: Dict) -> str:
        """Generate insights from features"""
        if not top_features:
            return "*No insights available*"
        
        lines = []
        
        sorted_features = sorted(top_features.items(), 
                                key=lambda x: x[1]['importance_score'], 
                                reverse=True)
        
        if sorted_features:
            top_feature, top_data = sorted_features[0]
            lines.append(f"**Dominant Feature:** `{top_feature}` is the most important predictor ")
            lines.append(f"with an importance score of {top_data['importance_score']:.4f}.\n\n")
        
        # Check concentration
        if len(sorted_features) >= 3:
            top_3_sum = sum(f[1]['importance_score'] for f in sorted_features[:3])
            total_sum = sum(f[1]['importance_score'] for f in sorted_features)
            concentration = top_3_sum / total_sum if total_sum > 0 else 0
            
            if concentration > 0.7:
                lines.append(f"**High Concentration:** Top 3 features account for {concentration*100:.0f}% of total importance.\n")
            else:
                lines.append(f"**Distributed Importance:** Features show relatively balanced importance distribution.\n")
        
        return "".join(lines)
    
    def _generate_feature_recommendations(self, top_features: Dict) -> str:
        """Generate feature-based recommendations"""
        recommendations = []
        
        if not top_features:
            return "*Insufficient data for recommendations*"
        
        sorted_features = sorted(top_features.items(), 
                                key=lambda x: x[1]['importance_score'], 
                                reverse=True)
        
        if sorted_features:
            top_3 = sorted_features[:3]
            feature_names = [f[0] for f in top_3]
            recommendations.append(f"1. **Focus on Top Drivers:** Prioritize data quality for {', '.join(feature_names)}")
        
        recommendations.append("2. **Feature Engineering:** Explore interactions between top features")
        recommendations.append("3. **Data Collection:** Ensure reliable measurement of high-importance features")
        
        return "\n".join(recommendations)
    
    # ========================================================================
    # UPLIFT ANALYSIS FILLERS
    # ========================================================================
    
    def _fill_uplift_analysis(self, template: str) -> str:
        """Fill uplift analysis placeholders"""
        filled = template
        
        overall_stats = self.computed_values.get('overall_stats', {})
        top_segments = self.computed_values.get('top_segments', [])
        
        # Overall stats
        if overall_stats:
            filled = filled.replace('{{MEAN_UPLIFT}}', f"{overall_stats.get('mean_uplift', 0):.4f}")
            filled = filled.replace('{{MEDIAN_UPLIFT}}', f"{overall_stats.get('median_uplift', 0):.4f}")
            filled = filled.replace('{{POSITIVE_UPLIFT_PCT}}', f"{overall_stats.get('positive_uplift_pct', 0):.1f}")
        
        # Top segments table
        segments_table = self._build_segments_table(top_segments)
        filled = filled.replace('{{TOP_SEGMENTS_TABLE}}', segments_table)
        
        # Segment analysis
        segment_analysis = self._generate_segment_analysis(overall_stats, top_segments)
        filled = filled.replace('{{SEGMENT_ANALYSIS}}', segment_analysis)
        
        # Recommendations
        recommendations = self._generate_uplift_recommendations(overall_stats, top_segments)
        filled = filled.replace('{{UPLIFT_RECOMMENDATIONS}}', recommendations)
        
        return filled
    
    def _build_segments_table(self, top_segments: List[Dict]) -> str:
        """Build top segments table"""
        if not top_segments:
            return "*No segment data available*"
        
        lines = []
        lines.append("| Rank | Segment | Uplift Score |\n")
        lines.append("|------|---------|-------------|\n")
        
        for rank, segment in enumerate(top_segments, 1):
            segment_id = segment.get('segment') or segment.get('entity_id', 'Unknown')
            uplift = segment.get('uplift_score', 0)
            
            emoji = "🎯" if rank == 1 else "⭐" if rank <= 3 else ""
            lines.append(f"| {rank} {emoji} | {segment_id} | {uplift:.4f} |\n")
        
        return "".join(lines)
    
    def _generate_segment_analysis(self, overall_stats: Dict, top_segments: List[Dict]) -> str:
        """Generate segment-level analysis"""
        lines = []
        
        if not overall_stats:
            return "*No statistical data available*"
        
        positive_pct = overall_stats.get('positive_uplift_pct', 0)
        
        if positive_pct > 70:
            lines.append("**Strong Positive Uplift:** The majority of segments show positive response to treatment, ")
            lines.append("indicating effective targeting strategy.\n\n")
        elif positive_pct > 40:
            lines.append("**Moderate Uplift:** Roughly half of segments respond positively, ")
            lines.append("suggesting room for refined targeting.\n\n")
        else:
            lines.append("**Low Positive Uplift:** Most segments show minimal or negative response, ")
            lines.append("indicating potential issues with campaign design or targeting.\n\n")
        
        if top_segments and len(top_segments) > 0:
            top_segment = top_segments[0]
            top_id = top_segment.get('segment') or top_segment.get('entity_id', 'Unknown')
            top_score = top_segment.get('uplift_score', 0)
            lines.append(f"The highest-performing segment (`{top_id}`) shows uplift of {top_score:.4f}, ")
            lines.append("representing the best targeting opportunity.\n")
        
        return "".join(lines)
    
    def _generate_uplift_recommendations(self, overall_stats: Dict, top_segments: List[Dict]) -> str:
        """Generate uplift recommendations"""
        recommendations = []
        
        if not overall_stats:
            return "*Insufficient data for recommendations*"
        
        positive_pct = overall_stats.get('positive_uplift_pct', 0)
        
        if positive_pct > 70:
            recommendations.append("1. **Scale Campaign:** High positive uplift justifies increased investment")
            recommendations.append("2. **Focus on Top Segments:** Prioritize segments with highest uplift scores")
        elif positive_pct > 40:
            recommendations.append("1. **Selective Targeting:** Focus resources on positive uplift segments only")
            recommendations.append("2. **Refine Messaging:** A/B test different approaches for low-uplift segments")
        else:
            recommendations.append("1. **Campaign Redesign:** Low uplift suggests fundamental issues with approach")
            recommendations.append("2. **Root Cause Analysis:** Investigate why treatment is ineffective")
        
        if top_segments:
            recommendations.append("3. **Lookalike Modeling:** Find similar segments to top performers")
        
        return "\n".join(recommendations)
    
    # ========================================================================
    # GENERAL FILLERS
    # ========================================================================
    
    def _fill_general(self, template: str) -> str:
        """Fill general placeholders"""
        filled = template
        
        data_shape = self.computed_values.get('data_shape', {})
        numerical_summary = self.computed_values.get('numerical_summary', {})
        
        # Data shape
        filled = filled.replace('{{NUM_ROWS}}', str(data_shape.get('rows', 0)))
        filled = filled.replace('{{NUM_COLUMNS}}', str(data_shape.get('columns', 0)))
        
        # Summary table
        summary_table = self._build_general_summary_table(numerical_summary)
        filled = filled.replace('{{SUMMARY_TABLE}}', summary_table)
        
        # Observations
        observations = self._generate_general_observations(numerical_summary)
        filled = filled.replace('{{OBSERVATIONS}}', observations)
        
        return filled
    
    def _build_general_summary_table(self, numerical_summary: Dict) -> str:
        """Build general summary table"""
        if not numerical_summary:
            return "*No numerical data available*"
        
        lines = []
        lines.append("| Column | Mean | Std Dev | Min | Max |\n")
        lines.append("|--------|------|---------|-----|-----|\n")
        
        for col, stats in numerical_summary.items():
            lines.append(f"| {col} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
        
        return "".join(lines)
    
    def _generate_general_observations(self, numerical_summary: Dict) -> str:
        """Generate general observations"""
        if not numerical_summary:
            return "*No observations available*"
        
        observations = []
        
        for col, stats in numerical_summary.items():
            if stats['std'] > stats['mean'] * 0.5:
                observations.append(f"- `{col}` shows high variability (std={stats['std']:.3f})")
        
        if not observations:
            observations.append("- Data appears relatively stable with low variability")
        
        return "\n".join(observations)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _cleanup_unfilled(self, text: str) -> str:
        """Remove or replace unfilled placeholders"""
        # Find all remaining placeholders
        placeholders = re.findall(r'\{\{([A-Z_]+)\}\}', text)
        
        for placeholder in placeholders:
            # Replace with placeholder name in brackets
            text = text.replace(f'{{{{{placeholder}}}}}', f'[{placeholder.replace("_", " ").title()}]')
        
        return text


def fill_template_placeholders(template: str, analysis_results: Dict[str, Any]) -> str:
    """
    Main entry point for filling placeholders
    
    Args:
        template: Template string with {{PLACEHOLDERS}}
        analysis_results: Analysis results with computed_values
    
    Returns:
        Filled template string
    """
    filler = PlaceholderFiller(template, analysis_results)
    return filler.fill()