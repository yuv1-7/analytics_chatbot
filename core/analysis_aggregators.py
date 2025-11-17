"""
Analysis aggregators that generate STORIES (not templates)
Stories describe what happened qualitatively for LLM to interpret
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class AnalysisAggregator:
    """Deterministic analysis that generates narrative stories"""
    
    def __init__(self, data: List[Dict], query_context: Dict):
        self.df = pd.DataFrame(data) if data else pd.DataFrame()
        self.context = query_context
        self.analysis_results = {}
    
    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis dispatcher
        
        Returns:
            {
                'analysis_type': str,
                'story': str,  # Qualitative narrative for LLM
                'computed_values': dict,  # Actual values for placeholder filling
                'story_elements': dict  # Structured story components
            }
        """
        if self.df.empty:
            return self._empty_result()
        
        comparison_type = self.context.get('comparison_type', 'general')
        use_case = self.context.get('use_case')
        
        # Route to appropriate analysis
        if comparison_type == 'performance':
            return self._analyze_model_performance()
        elif comparison_type == 'drift':
            return self._analyze_drift()
        elif comparison_type == 'ensemble_vs_base':
            return self._analyze_ensemble_vs_base()
        elif comparison_type == 'feature_importance':
            return self._analyze_feature_importance()
        elif 'uplift' in str(use_case).lower():
            return self._analyze_uplift()
        else:
            return self._analyze_general()
    
    # ========================================================================
    # PERFORMANCE COMPARISON ANALYSIS
    # ========================================================================
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """
        Analyze model performance and generate story
        """
        # Compute all metrics
        metrics_summary = self._compute_metrics_summary()
        rankings = self._compute_rankings(metrics_summary)
        
        # Generate story
        story = self._generate_performance_story(metrics_summary, rankings)
        
        # Structure story elements for LLM
        story_elements = self._extract_performance_story_elements(metrics_summary, rankings)
        
        # Store computed values for placeholders
        computed_values = self._build_computed_values(metrics_summary, rankings)
        
        return {
            'analysis_type': 'performance_comparison',
            'story': story,
            'story_elements': story_elements,
            'computed_values': computed_values,
            'models_analyzed': list(self.df['model_name'].unique()) if 'model_name' in self.df.columns else []
        }
    
    def _compute_metrics_summary(self) -> Dict[str, Any]:
        """Compute metrics (same as before)"""
        metrics_summary = {}
        
        if 'metric_name' in self.df.columns:
            # Long format
            for metric in self.df['metric_name'].unique():
                metric_df = self.df[self.df['metric_name'] == metric]
                
                if 'model_name' in metric_df.columns:
                    metric_summary = {}
                    for model in metric_df['model_name'].unique():
                        values = metric_df[metric_df['model_name'] == model]['metric_value']
                        metric_summary[model] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()) if len(values) > 1 else 0.0,
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'count': int(len(values))
                        }
                    metrics_summary[metric] = metric_summary
        else:
            # Wide format
            metric_cols = [col for col in self.df.columns 
                           if any(m in col.lower() for m in ['rmse', 'mae', 'r2', 'auc', 'accuracy'])]
            
            for metric_col in metric_cols:
                metric_name = metric_col.lower().replace('test_', '').replace('val_', '')
                
                if 'model_name' in self.df.columns:
                    metric_summary = {}
                    for model in self.df['model_name'].unique():
                        values = self.df[self.df['model_name'] == model][metric_col].dropna()
                        if len(values) > 0:
                            metric_summary[model] = {
                                'mean': float(values.mean()),
                                'std': float(values.std()) if len(values) > 1 else 0.0,
                                'min': float(values.min()),
                                'max': float(values.max()),
                                'count': int(len(values))
                            }
                    metrics_summary[metric_name] = metric_summary
        
        return metrics_summary
    
    def _compute_rankings(self, metrics_summary: Dict) -> Dict[str, List]:
        """Compute rankings (same as before)"""
        rankings = {}
        lower_better = ['rmse', 'mae', 'mse', 'mape']
        
        for metric, model_values in metrics_summary.items():
            metric_lower = metric.lower()
            model_means = [(model, vals['mean']) for model, vals in model_values.items()]
            
            if any(lb in metric_lower for lb in lower_better):
                ranked = sorted(model_means, key=lambda x: x[1])
            else:
                ranked = sorted(model_means, key=lambda x: x[1], reverse=True)
            
            rankings[metric] = ranked
        
        return rankings
    
    def _generate_performance_story(self, metrics_summary: Dict, rankings: Dict) -> str:
        """
        Generate a rich qualitative story for LLM
        
        This is the KEY method - tells what happened without numbers
        """
        story_parts = []
        
        # Introduction
        num_models = len(set(model for metric in metrics_summary.values() for model in metric))
        num_metrics = len(metrics_summary)
        story_parts.append(f"ANALYSIS_CONTEXT: Compared {num_models} models across {num_metrics} performance metrics.")
        
        # For each metric, tell the story
        for metric, model_values in metrics_summary.items():
            if metric not in rankings or len(rankings[metric]) < 2:
                continue
            
            ranked = rankings[metric]
            winner = ranked[0][0]
            winner_val = ranked[0][1]
            
            # Calculate relative performance
            story_parts.append(f"\nMETRIC_STORY for {metric.upper()}:")
            
            # Winner announcement
            story_parts.append(f"  - WINNER: {winner} ranked first")
            
            # Performance gaps
            if len(ranked) >= 2:
                second = ranked[1][0]
                second_val = ranked[1][1]
                gap_pct = abs((winner_val - second_val) / second_val * 100)
                
                if gap_pct > 15:
                    story_parts.append(f"  - GAP_SIZE: LARGE - {winner} significantly outperforms {second} by over 15%")
                elif gap_pct > 5:
                    story_parts.append(f"  - GAP_SIZE: MODERATE - {winner} moderately outperforms {second} by 5-15%")
                else:
                    story_parts.append(f"  - GAP_SIZE: SMALL - {winner} and {second} are very close (under 5% difference)")
            
            # Consistency analysis
            winner_std = model_values[winner]['std']
            winner_mean = model_values[winner]['mean']
            cv = winner_std / winner_mean if winner_mean != 0 else 0
            
            if cv < 0.1:
                story_parts.append(f"  - CONSISTENCY: {winner} shows HIGH consistency (very stable predictions)")
            elif cv < 0.3:
                story_parts.append(f"  - CONSISTENCY: {winner} shows MODERATE consistency")
            else:
                story_parts.append(f"  - CONSISTENCY: {winner} shows HIGH variability (unstable predictions)")
            
            # Relative positioning of all models
            if len(ranked) >= 3:
                last = ranked[-1][0]
                last_val = ranked[-1][1]
                range_pct = abs((winner_val - last_val) / last_val * 100)
                
                if range_pct > 30:
                    story_parts.append(f"  - SPREAD: WIDE - Performance varies greatly across models (>30% range)")
                elif range_pct > 10:
                    story_parts.append(f"  - SPREAD: MODERATE - Models show meaningful differences (10-30% range)")
                else:
                    story_parts.append(f"  - SPREAD: NARROW - All models perform similarly (<10% range)")
        
        # Cross-metric consistency
        if len(rankings) >= 2:
            # Check if same model wins across metrics
            winners = [ranked[0][0] for ranked in rankings.values()]
            winner_counts = {model: winners.count(model) for model in set(winners)}
            dominant_model = max(winner_counts, key=winner_counts.get)
            dominance = winner_counts[dominant_model] / len(rankings)
            
            story_parts.append(f"\nCROSS_METRIC_ANALYSIS:")
            if dominance >= 0.8:
                story_parts.append(f"  - DOMINANCE: STRONG - {dominant_model} wins on most metrics (clear overall winner)")
            elif dominance >= 0.5:
                story_parts.append(f"  - DOMINANCE: MODERATE - {dominant_model} wins on half the metrics (mixed results)")
            else:
                story_parts.append(f"  - DOMINANCE: NONE - No single model dominates (highly context-dependent performance)")
        
        # Actionability assessment
        story_parts.append(f"\nACTIONABILITY:")
        best_models = set(ranked[0][0] for ranked in rankings.values())
        if len(best_models) == 1:
            story_parts.append(f"  - CLARITY: HIGH - Clear single best model for deployment")
        else:
            story_parts.append(f"  - CLARITY: LOW - Multiple models excel at different metrics, requires business prioritization")
        
        return "\n".join(story_parts)
    
    def _extract_performance_story_elements(self, metrics_summary: Dict, rankings: Dict) -> Dict:
        """
        Extract structured story elements for more granular LLM control
        """
        elements = {
            'winners_by_metric': {},
            'performance_gaps': {},
            'consistency_levels': {},
            'overall_winner': None,
            'decision_clarity': 'unclear'
        }
        
        # Extract winners and gaps for each metric
        for metric, ranked in rankings.items():
            if len(ranked) >= 1:
                winner = ranked[0][0]
                elements['winners_by_metric'][metric] = winner
                
                if len(ranked) >= 2:
                    gap = abs((ranked[0][1] - ranked[1][1]) / ranked[1][1] * 100)
                    elements['performance_gaps'][metric] = {
                        'winner': winner,
                        'runner_up': ranked[1][0],
                        'gap_category': 'large' if gap > 15 else 'moderate' if gap > 5 else 'small'
                    }
            
            # Consistency
            if winner in metrics_summary.get(metric, {}):
                winner_data = metrics_summary[metric][winner]
                cv = winner_data['std'] / winner_data['mean'] if winner_data['mean'] != 0 else 0
                elements['consistency_levels'][metric] = {
                    'model': winner,
                    'level': 'high' if cv < 0.1 else 'moderate' if cv < 0.3 else 'low'
                }
        
        # Overall winner
        winners = list(elements['winners_by_metric'].values())
        if winners:
            winner_counts = {m: winners.count(m) for m in set(winners)}
            overall_winner = max(winner_counts, key=winner_counts.get)
            dominance = winner_counts[overall_winner] / len(winners)
            
            elements['overall_winner'] = {
                'model': overall_winner,
                'wins': winner_counts[overall_winner],
                'total_metrics': len(winners),
                'dominance': 'strong' if dominance >= 0.8 else 'moderate' if dominance >= 0.5 else 'weak'
            }
            
            if dominance >= 0.8:
                elements['decision_clarity'] = 'high'
            elif dominance >= 0.5:
                elements['decision_clarity'] = 'moderate'
            else:
                elements['decision_clarity'] = 'low'
        
        return elements
    
    def _build_computed_values(self, metrics_summary: Dict, rankings: Dict) -> Dict:
        """Store actual computed values for placeholder filling"""
        return {
            'metrics_summary': metrics_summary,
            'rankings': rankings,
            'timestamp': datetime.now().isoformat()
        }
    
    # ========================================================================
    # DRIFT ANALYSIS
    # ========================================================================
    
    def _analyze_drift(self) -> Dict[str, Any]:
        """Analyze drift and generate story"""
        drift_summary = self._compute_drift_summary()
        story = self._generate_drift_story(drift_summary)
        story_elements = self._extract_drift_story_elements(drift_summary)
        computed_values = {'drift_summary': drift_summary}
        
        return {
            'analysis_type': 'drift_analysis',
            'story': story,
            'story_elements': story_elements,
            'computed_values': computed_values
        }
    
    def _compute_drift_summary(self) -> Dict[str, Any]:
        """Compute drift metrics"""
        drift_summary = {}
        
        if 'drift_detected' not in self.df.columns:
            return drift_summary
        
        if 'model_name' in self.df.columns and 'drift_score' in self.df.columns:
            for model in self.df['model_name'].unique():
                model_df = self.df[self.df['model_name'] == model]
                drift_summary[model] = {
                    'mean_drift_score': float(model_df['drift_score'].mean()),
                    'max_drift_score': float(model_df['drift_score'].max()),
                    'drift_detected': bool(model_df['drift_detected'].any()),
                    'drift_type': model_df['drift_type'].mode()[0] if 'drift_type' in model_df.columns else 'unknown'
                }
        
        return drift_summary
    
    def _generate_drift_story(self, drift_summary: Dict) -> str:
        """Generate drift narrative story"""
        if not drift_summary:
            return "NO_DRIFT_DATA: No drift detection data available for analysis."
        
        story_parts = []
        
        total_models = len(drift_summary)
        models_with_drift = [m for m, d in drift_summary.items() if d['drift_detected']]
        drift_rate = len(models_with_drift) / total_models if total_models > 0 else 0
        
        story_parts.append(f"DRIFT_OVERVIEW: Analyzed {total_models} models for drift")
        story_parts.append(f"  - DRIFT_PREVALENCE: {len(models_with_drift)} models show drift ({drift_rate*100:.0f}% rate)")
        
        # Severity assessment
        if drift_rate > 0.5:
            story_parts.append(f"  - SEVERITY: CRITICAL - Majority of models experiencing drift (systemic issue)")
        elif drift_rate > 0.2:
            story_parts.append(f"  - SEVERITY: MODERATE - Significant portion of models affected")
        elif drift_rate > 0:
            story_parts.append(f"  - SEVERITY: LOW - Isolated drift cases")
        else:
            story_parts.append(f"  - SEVERITY: NONE - All models stable")
        
        # Individual model stories
        if models_with_drift:
            story_parts.append(f"\nMODEL_DRIFT_DETAILS:")
            
            for model in models_with_drift:
                drift_data = drift_summary[model]
                score = drift_data['mean_drift_score']
                drift_type = drift_data.get('drift_type', 'unknown')
                
                if score > 0.15:
                    urgency = "URGENT"
                elif score > 0.10:
                    urgency = "MODERATE"
                else:
                    urgency = "LOW"
                
                story_parts.append(f"  - {model}:")
                story_parts.append(f"      URGENCY: {urgency}")
                story_parts.append(f"      TYPE: {drift_type}")
                
                if urgency == "URGENT":
                    story_parts.append(f"      ACTION: Immediate retraining required")
                elif urgency == "MODERATE":
                    story_parts.append(f"      ACTION: Schedule retraining within next cycle")
                else:
                    story_parts.append(f"      ACTION: Monitor closely, no immediate action")
        
        # Root cause speculation
        story_parts.append(f"\nPOSSIBLE_CAUSES:")
        drift_types = [d.get('drift_type', 'unknown') for d in drift_summary.values() if d['drift_detected']]
        if drift_types:
            most_common_type = max(set(drift_types), key=drift_types.count)
            story_parts.append(f"  - DOMINANT_TYPE: {most_common_type}")
            
            if most_common_type == 'concept_drift':
                story_parts.append(f"  - LIKELY_CAUSE: Relationship between features and target has changed")
            elif most_common_type == 'data_drift':
                story_parts.append(f"  - LIKELY_CAUSE: Input data distributions have shifted")
            elif most_common_type == 'performance_drift':
                story_parts.append(f"  - LIKELY_CAUSE: Model accuracy degrading over time")
        
        return "\n".join(story_parts)
    
    def _extract_drift_story_elements(self, drift_summary: Dict) -> Dict:
        """Extract structured drift story elements"""
        models_with_drift = [m for m, d in drift_summary.items() if d['drift_detected']]
        
        return {
            'total_models': len(drift_summary),
            'models_with_drift': models_with_drift,
            'drift_rate': len(models_with_drift) / len(drift_summary) if drift_summary else 0,
            'severity': 'critical' if len(models_with_drift) / len(drift_summary) > 0.5 else 'moderate' if len(models_with_drift) / len(drift_summary) > 0.2 else 'low',
            'urgent_models': [m for m, d in drift_summary.items() if d.get('mean_drift_score', 0) > 0.15]
        }
    
    # ========================================================================
    # ENSEMBLE VS BASE ANALYSIS
    # ========================================================================
    
    def _analyze_ensemble_vs_base(self) -> Dict[str, Any]:
        """Analyze ensemble vs base and generate story"""
        comparison_data = self._compute_ensemble_comparison()
        story = self._generate_ensemble_story(comparison_data)
        story_elements = self._extract_ensemble_story_elements(comparison_data)
        computed_values = comparison_data
        
        return {
            'analysis_type': 'ensemble_vs_base',
            'story': story,
            'story_elements': story_elements,
            'computed_values': computed_values
        }
    
    def _compute_ensemble_comparison(self) -> Dict[str, Any]:
        """Compute ensemble vs base metrics"""
        if 'model_type' not in self.df.columns:
            return {}
        
        ensemble_df = self.df[self.df['model_type'] == 'ensemble']
        base_df = self.df[self.df['model_type'] == 'base_model']
        
        comparison = {
            'ensemble_metrics': {},
            'base_metrics': {},
            'advantages': {}
        }
        
        if 'metric_name' in self.df.columns:
            for metric in self.df['metric_name'].unique():
                ens_vals = ensemble_df[ensemble_df['metric_name'] == metric]['metric_value']
                base_vals = base_df[base_df['metric_name'] == metric]['metric_value']
                
                if len(ens_vals) > 0 and len(base_vals) > 0:
                    ens_mean = float(ens_vals.mean())
                    base_mean = float(base_vals.mean())
                    
                    comparison['ensemble_metrics'][metric] = ens_mean
                    comparison['base_metrics'][metric] = base_mean
                    
                    diff = ens_mean - base_mean
                    pct_change = (diff / base_mean * 100) if base_mean != 0 else 0
                    
                    is_better = self._is_ensemble_better(metric, ens_mean, base_mean)
                    
                    comparison['advantages'][metric] = {
                        'absolute_diff': diff,
                        'percent_change': pct_change,
                        'ensemble_better': is_better
                    }
        
        return comparison
    
    def _is_ensemble_better(self, metric: str, ens_val: float, base_val: float) -> bool:
        """Check if ensemble is better"""
        lower_better = ['rmse', 'mae', 'mse', 'mape']
        if any(lb in metric.lower() for lb in lower_better):
            return ens_val < base_val
        return ens_val > base_val
    
    def _generate_ensemble_story(self, comparison_data: Dict) -> str:
        """Generate ensemble comparison story"""
        if not comparison_data or not comparison_data.get('advantages'):
            return "NO_COMPARISON_DATA: Insufficient data to compare ensemble and base models."
        
        story_parts = []
        advantages = comparison_data['advantages']
        
        # Calculate win rate
        wins = sum(1 for adv in advantages.values() if adv['ensemble_better'])
        total = len(advantages)
        win_rate = wins / total if total > 0 else 0
        
        story_parts.append(f"ENSEMBLE_COMPARISON:")
        story_parts.append(f"  - WIN_RATE: Ensemble wins on {wins}/{total} metrics ({win_rate*100:.0f}%)")
        
        # Overall verdict
        if win_rate >= 0.8:
            story_parts.append(f"  - VERDICT: STRONG_ENSEMBLE_ADVANTAGE - Ensemble clearly superior")
        elif win_rate >= 0.5:
            story_parts.append(f"  - VERDICT: MIXED_PERFORMANCE - Ensemble better on some metrics, worse on others")
        elif win_rate > 0:
            story_parts.append(f"  - VERDICT: BASE_MODEL_ADVANTAGE - Base models generally outperform")
        else:
            story_parts.append(f"  - VERDICT: ENSEMBLE_FAILURE - Ensemble underperforms on all metrics")
        
        # Magnitude analysis
        story_parts.append(f"\nPERFORMANCE_MAGNITUDE:")
        for metric, adv in advantages.items():
            pct = abs(adv['percent_change'])
            better = "ENSEMBLE_WINS" if adv['ensemble_better'] else "BASE_WINS"
            
            if pct > 15:
                magnitude = "LARGE"
            elif pct > 5:
                magnitude = "MODERATE"
            else:
                magnitude = "SMALL"
            
            story_parts.append(f"  - {metric}: {better} by {magnitude} margin ({pct:.1f}%)")
        
        # Diagnosis
        story_parts.append(f"\nDIAGNOSIS:")
        if win_rate >= 0.8:
            story_parts.append(f"  - CAUSE: Effective model diversity and meta-learner optimization")
            story_parts.append(f"  - RECOMMENDATION: Deploy ensemble - complexity justified")
        elif win_rate >= 0.5:
            story_parts.append(f"  - CAUSE: Ensemble excels in some areas but not others")
            story_parts.append(f"  - RECOMMENDATION: Selective deployment or ensemble refinement")
        else:
            story_parts.append(f"  - CAUSE: Likely insufficient diversity or meta-learner overfitting")
            story_parts.append(f"  - RECOMMENDATION: Use best base model; redesign ensemble")
        
        return "\n".join(story_parts)
    
    def _extract_ensemble_story_elements(self, comparison_data: Dict) -> Dict:
        """Extract structured ensemble story elements"""
        if not comparison_data or not comparison_data.get('advantages'):
            return {}
        
        advantages = comparison_data['advantages']
        wins = sum(1 for adv in advantages.values() if adv['ensemble_better'])
        total = len(advantages)
        
        return {
            'win_rate': wins / total if total > 0 else 0,
            'wins': wins,
            'total_metrics': total,
            'verdict': 'strong_advantage' if wins/total >= 0.8 else 'mixed' if wins/total >= 0.5 else 'underperforms',
            'significant_wins': [m for m, adv in advantages.items() if adv['ensemble_better'] and abs(adv['percent_change']) > 10],
            'significant_losses': [m for m, adv in advantages.items() if not adv['ensemble_better'] and abs(adv['percent_change']) > 10]
        }
    
    # ========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance and generate story"""
        feature_data = self._compute_feature_data()
        story = self._generate_feature_story(feature_data)
        story_elements = self._extract_feature_story_elements(feature_data)
        computed_values = feature_data
        
        return {
            'analysis_type': 'feature_importance',
            'story': story,
            'story_elements': story_elements,
            'computed_values': computed_values
        }
    
    def _compute_feature_data(self) -> Dict[str, Any]:
        """Compute feature importance data"""
        if 'feature_name' not in self.df.columns or 'importance_score' not in self.df.columns:
            return {}
        
        top_features = self.df.nlargest(10, 'importance_score')
        
        feature_data = {
            'top_features': {},
            'total_features': len(self.df)
        }
        
        for _, row in top_features.iterrows():
            feature_data['top_features'][row['feature_name']] = {
                'importance_score': float(row['importance_score']),
                'rank': int(row['rank']) if 'rank' in row else None
            }
        
        return feature_data
    
    def _generate_feature_story(self, feature_data: Dict) -> str:
        """Generate feature importance story"""
        if not feature_data or not feature_data.get('top_features'):
            return "NO_FEATURE_DATA: No feature importance data available."
        
        story_parts = []
        top_features = feature_data['top_features']
        total_features = feature_data['total_features']
        
        story_parts.append(f"FEATURE_IMPORTANCE_ANALYSIS:")
        story_parts.append(f"  - TOTAL_FEATURES: {total_features} features analyzed")
        story_parts.append(f"  - TOP_FEATURES_SHOWN: {len(top_features)} most important")
        
        # Analyze concentration
        sorted_features = sorted(top_features.items(), key=lambda x: x[1]['importance_score'], reverse=True)
        
        if len(sorted_features) >= 3:
            top_3_sum = sum(f[1]['importance_score'] for f in sorted_features[:3])
            total_sum = sum(f[1]['importance_score'] for f in sorted_features)
            concentration = top_3_sum / total_sum if total_sum > 0 else 0
            
            story_parts.append(f"\nIMPORTANCE_DISTRIBUTION:")
            if concentration > 0.7:
                story_parts.append(f"  - CONCENTRATION: HIGH - Top 3 features dominate (over 70% of importance)")
                story_parts.append(f"  - IMPLICATION: Model heavily reliant on few features")
            elif concentration > 0.4:
                story_parts.append(f"  - CONCENTRATION: MODERATE - Top 3 features significant but not dominant")
                story_parts.append(f"  - IMPLICATION: Balanced feature contribution")
            else:
                story_parts.append(f"  - CONCENTRATION: LOW - Importance widely distributed")
                story_parts.append(f"  - IMPLICATION: Model uses many features equally")
        
        # Top feature analysis
        if sorted_features:
            top_feature, top_data = sorted_features[0]
            story_parts.append(f"\nTOP_FEATURE_ANALYSIS:")
            story_parts.append(f"  - DOMINANT_FEATURE: {top_feature}")
            
            if len(sorted_features) >= 2:
                second_feature, second_data = sorted_features[1]
                ratio = top_data['importance_score'] / second_data['importance_score']
                
                if ratio > 2.0:
                    story_parts.append(f"  - DOMINANCE_LEVEL: EXTREME - Top feature 2x more important than second")
                elif ratio > 1.5:
                    story_parts.append(f"  - DOMINANCE_LEVEL: STRONG - Clear importance leader")
                else:
                    story_parts.append(f"  - DOMINANCE_LEVEL: MODERATE - Top features relatively balanced")
        
        # Business implications
        story_parts.append(f"\nBUSINESS_IMPLICATIONS:")
        if 'historical' in str(top_feature).lower() or 'lag' in str(top_feature).lower():
            story_parts.append(f"  - KEY_DRIVER: Historical patterns - Past behavior strongly predicts future")
        elif 'volume' in str(top_feature).lower() or 'patient' in str(top_feature).lower():
            story_parts.append(f"  - KEY_DRIVER: Volume metrics - Scale drives predictions")
        elif 'specialty' in str(top_feature).lower():
            story_parts.append(f"  - KEY_DRIVER: Provider characteristics - Who matters more than what")
        
        return "\n".join(story_parts)
    
    def _extract_feature_story_elements(self, feature_data: Dict) -> Dict:
        """Extract structured feature story elements"""
        if not feature_data or not feature_data.get('top_features'):
            return {}
        
        top_features = feature_data['top_features']
        sorted_features = sorted(top_features.items(), key=lambda x: x[1]['importance_score'], reverse=True)
        
        elements = {
            'total_features': feature_data['total_features'],
            'top_count': len(top_features),
            'dominant_feature': sorted_features[0][0] if sorted_features else None
        }
        
        # Concentration analysis
        if len(sorted_features) >= 3:
            top_3_sum = sum(f[1]['importance_score'] for f in sorted_features[:3])
            total_sum = sum(f[1]['importance_score'] for f in sorted_features)
            concentration = top_3_sum / total_sum if total_sum > 0 else 0
            
            elements['concentration'] = 'high' if concentration > 0.7 else 'moderate' if concentration > 0.4 else 'low'
            elements['top_3_contribution'] = concentration
        
        # Dominance level
        if len(sorted_features) >= 2:
            ratio = sorted_features[0][1]['importance_score'] / sorted_features[1][1]['importance_score']
            elements['dominance_level'] = 'extreme' if ratio > 2.0 else 'strong' if ratio > 1.5 else 'moderate'
        
        return elements
    
    # ========================================================================
    # UPLIFT ANALYSIS
    # ========================================================================
    
    def _analyze_uplift(self) -> Dict[str, Any]:
        """Analyze uplift and generate story"""
        uplift_data = self._compute_uplift_data()
        story = self._generate_uplift_story(uplift_data)
        story_elements = self._extract_uplift_story_elements(uplift_data)
        computed_values = uplift_data
        
        return {
            'analysis_type': 'uplift_analysis',
            'story': story,
            'story_elements': story_elements,
            'computed_values': computed_values
        }
    
    def _compute_uplift_data(self) -> Dict[str, Any]:
        """Compute uplift metrics"""
        if 'uplift_score' not in self.df.columns:
            return {}
        
        uplift_data = {
            'overall_stats': {
                'mean_uplift': float(self.df['uplift_score'].mean()),
                'median_uplift': float(self.df['uplift_score'].median()),
                'positive_rate': float((self.df['uplift_score'] > 0).sum() / len(self.df) * 100)
            },
            'top_segments': []
        }
        
        # Top segments
        if 'segment' in self.df.columns or 'entity_id' in self.df.columns:
            segment_col = 'segment' if 'segment' in self.df.columns else 'entity_id'
            top_5 = self.df.nlargest(5, 'uplift_score')
            
            uplift_data['top_segments'] = [
                {
                    'segment': row[segment_col],
                    'uplift_score': float(row['uplift_score'])
                }
                for _, row in top_5.iterrows()
            ]
        
        return uplift_data
    
    def _generate_uplift_story(self, uplift_data: Dict) -> str:
        """Generate uplift analysis story"""
        if not uplift_data or not uplift_data.get('overall_stats'):
            return "NO_UPLIFT_DATA: No uplift modeling data available."
        
        story_parts = []
        stats = uplift_data['overall_stats']
        positive_rate = stats['positive_rate']
        
        story_parts.append(f"UPLIFT_ANALYSIS:")
        story_parts.append(f"  - POSITIVE_UPLIFT_RATE: {positive_rate:.0f}% of segments show positive response")
        
        # Overall assessment
        if positive_rate > 70:
            story_parts.append(f"  - CAMPAIGN_EFFECTIVENESS: STRONG - Majority respond positively")
            story_parts.append(f"  - BUSINESS_IMPACT: High ROI expected from targeting")
        elif positive_rate > 40:
            story_parts.append(f"  - CAMPAIGN_EFFECTIVENESS: MODERATE - Mixed response across segments")
            story_parts.append(f"  - BUSINESS_IMPACT: Selective targeting recommended")
        else:
            story_parts.append(f"  - CAMPAIGN_EFFECTIVENESS: LOW - Most segments unresponsive or negative")
            story_parts.append(f"  - BUSINESS_IMPACT: Campaign redesign needed")
        
        # Distribution analysis
        mean_uplift = stats['mean_uplift']
        median_uplift = stats['median_uplift']
        
        story_parts.append(f"\nUPLIFT_DISTRIBUTION:")
        if mean_uplift > median_uplift * 1.2:
            story_parts.append(f"  - SHAPE: RIGHT_SKEWED - Few high performers pulling average up")
            story_parts.append(f"  - STRATEGY: Focus on identifying high-uplift segments")
        elif median_uplift > mean_uplift * 1.2:
            story_parts.append(f"  - SHAPE: LEFT_SKEWED - Few poor performers pulling average down")
            story_parts.append(f"  - STRATEGY: Avoid negative-uplift segments")
        else:
            story_parts.append(f"  - SHAPE: SYMMETRIC - Relatively normal distribution")
            story_parts.append(f"  - STRATEGY: Standard targeting approach")
        
        # Top performers
        if uplift_data.get('top_segments'):
            story_parts.append(f"\nTOP_PERFORMERS:")
            top_segment = uplift_data['top_segments'][0]
            story_parts.append(f"  - BEST_SEGMENT: {top_segment['segment']}")
            story_parts.append(f"  - OPPORTUNITY: Highest incremental response potential")
            
            if len(uplift_data['top_segments']) >= 3:
                story_parts.append(f"  - TOP_3_PATTERN: Identify common characteristics for lookalike modeling")
        
        # Actionability
        story_parts.append(f"\nRECOMMENDED_ACTION:")
        if positive_rate > 70:
            story_parts.append(f"  - SCALE: Expand campaign to positive-uplift segments")
            story_parts.append(f"  - INVEST: Increase budget allocation")
        elif positive_rate > 40:
            story_parts.append(f"  - REFINE: Target only positive-uplift segments")
            story_parts.append(f"  - TEST: A/B test different approaches for neutral segments")
        else:
            story_parts.append(f"  - REDESIGN: Fundamental campaign changes needed")
            story_parts.append(f"  - INVESTIGATE: Root cause analysis of low uplift")
        
        return "\n".join(story_parts)
    
    def _extract_uplift_story_elements(self, uplift_data: Dict) -> Dict:
        """Extract structured uplift story elements"""
        if not uplift_data or not uplift_data.get('overall_stats'):
            return {}
        
        stats = uplift_data['overall_stats']
        positive_rate = stats['positive_rate']
        
        return {
            'positive_rate': positive_rate,
            'effectiveness': 'strong' if positive_rate > 70 else 'moderate' if positive_rate > 40 else 'low',
            'top_segment': uplift_data['top_segments'][0]['segment'] if uplift_data.get('top_segments') else None,
            'distribution_shape': self._determine_distribution_shape(stats['mean_uplift'], stats['median_uplift']),
            'recommended_action': 'scale' if positive_rate > 70 else 'refine' if positive_rate > 40 else 'redesign'
        }
    
    def _determine_distribution_shape(self, mean: float, median: float) -> str:
        """Determine distribution shape from mean/median"""
        if mean > median * 1.2:
            return 'right_skewed'
        elif median > mean * 1.2:
            return 'left_skewed'
        else:
            return 'symmetric'
    
    # ========================================================================
    # GENERAL ANALYSIS
    # ========================================================================
    
    def _analyze_general(self) -> Dict[str, Any]:
        """General analysis and story"""
        summary = self._compute_general_summary()
        story = self._generate_general_story(summary)
        story_elements = {'data_shape': summary['data_shape']}
        computed_values = summary
        
        return {
            'analysis_type': 'general',
            'story': story,
            'story_elements': story_elements,
            'computed_values': computed_values
        }
    
    def _compute_general_summary(self) -> Dict[str, Any]:
        """Compute general summary statistics"""
        summary = {
            'data_shape': {
                'rows': len(self.df),
                'columns': len(self.df.columns)
            },
            'columns': list(self.df.columns),
            'numerical_summary': {}
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numerical_summary'][col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        
        return summary
    
    def _generate_general_story(self, summary: Dict) -> str:
        """Generate general data story"""
        story_parts = []
        
        story_parts.append(f"DATA_SUMMARY:")
        story_parts.append(f"  - DATASET_SIZE: {summary['data_shape']['rows']} records")
        story_parts.append(f"  - COLUMNS: {summary['data_shape']['columns']} fields")
        
        if summary['numerical_summary']:
            story_parts.append(f"\nNUMERICAL_CHARACTERISTICS:")
            for col, stats in list(summary['numerical_summary'].items())[:5]:
                cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
                variability = "HIGH" if cv > 0.5 else "MODERATE" if cv > 0.2 else "LOW"
                story_parts.append(f"  - {col}: {variability} variability")
        
        return "\n".join(story_parts)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'analysis_type': 'empty',
            'story': "NO_DATA: No data available for analysis.",
            'story_elements': {},
            'computed_values': {}
        }


def analyze_data(data: List[Dict], query_context: Dict) -> Dict[str, Any]:
    """
    Main entry point for story-based analysis
    
    Args:
        data: Raw SQL query results
        query_context: Parsed intent and context
    
    Returns:
        Analysis with rich narrative story + computed values
    """
    aggregator = AnalysisAggregator(data, query_context)
    return aggregator.analyze()