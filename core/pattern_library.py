from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class Pattern:
    """Represents a detected pattern"""
    pattern_type: str
    severity: str  # LOW, MODERATE, HIGH, CRITICAL
    description: str
    business_impact: str
    recommended_action: str
    evidence: List[str]


class PatternLibrary:
    """Library of recognized patterns and their business interpretations"""
    
    @staticmethod
    def detect_performance_patterns(metrics_summary: Dict, rankings: Dict) -> List[Pattern]:
        """Detect all relevant patterns in performance data"""
        patterns = []
        
        # Pattern 1: Dominant Model
        patterns.extend(PatternLibrary._detect_dominant_model(rankings))
        
        # Pattern 2: Tight Race
        patterns.extend(PatternLibrary._detect_tight_race(rankings))
        
        # Pattern 3: Instability
        patterns.extend(PatternLibrary._detect_instability(metrics_summary))
        
        # Pattern 4: Metric Disagreement
        patterns.extend(PatternLibrary._detect_metric_disagreement(rankings))
        
        # Pattern 5: Ensemble Potential
        patterns.extend(PatternLibrary._detect_ensemble_potential(rankings, metrics_summary))
        
        return patterns
    
    @staticmethod
    def _detect_dominant_model(rankings: Dict) -> List[Pattern]:
        """Detect if one model dominates"""
        patterns = []
        
        winners = [ranked[0][0] for ranked in rankings.values()]
        winner_counts = {model: winners.count(model) for model in set(winners)}
        dominant_model = max(winner_counts, key=winner_counts.get)
        dominance_rate = winner_counts[dominant_model] / len(rankings)
        
        if dominance_rate >= 0.8:
            patterns.append(Pattern(
                pattern_type="DOMINANT_MODEL",
                severity="HIGH",
                description=f"{dominant_model} wins on {dominance_rate*100:.0f}% of metrics",
                business_impact="Clear deployment choice; minimal decision complexity",
                recommended_action=f"Deploy {dominant_model} to production; monitor performance",
                evidence=[
                    f"Wins {winner_counts[dominant_model]}/{len(rankings)} metrics",
                    "Consistent advantage across evaluation dimensions",
                    "Low risk of performance regression"
                ]
            ))
        
        return patterns
    
    @staticmethod
    def _detect_tight_race(rankings: Dict) -> List[Pattern]:
        """Detect tight races between models"""
        patterns = []
        
        close_races = []
        for metric, ranked in rankings.items():
            if len(ranked) >= 2:
                top_val = ranked[0][1]
                second_val = ranked[1][1]
                gap = abs((top_val - second_val) / second_val)
                
                if gap < 0.03:  # Within 3%
                    close_races.append({
                        'metric': metric,
                        'winner': ranked[0][0],
                        'runner_up': ranked[1][0],
                        'gap_percent': gap * 100
                    })
        
        if len(close_races) > len(rankings) / 2:
            patterns.append(Pattern(
                pattern_type="TIGHT_RACE",
                severity="MODERATE",
                description=f"{len(close_races)} metrics show extremely close competition",
                business_impact="Differences may not be statistically significant; decision factors beyond performance",
                recommended_action="Consider non-performance factors: cost, maintenance, interpretability",
                evidence=[
                    f"{len(close_races)}/{len(rankings)} metrics within 3% margin",
                    "Statistical uncertainty in rankings",
                    "Performance alone insufficient for decision"
                ]
            ))
        
        return patterns
    
    @staticmethod
    def _detect_instability(metrics_summary: Dict) -> List[Pattern]:
        """Detect prediction instability"""
        patterns = []
        
        unstable_models = []
        for metric, model_data in metrics_summary.items():
            for model, stats in model_data.items():
                cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
                if cv > 0.3:
                    unstable_models.append({
                        'model': model,
                        'metric': metric,
                        'cv': cv
                    })
        
        if unstable_models:
            patterns.append(Pattern(
                pattern_type="HIGH_INSTABILITY",
                severity="HIGH" if len(unstable_models) > 3 else "MODERATE",
                description=f"{len(unstable_models)} model-metric combinations show high variance",
                business_impact="Unpredictable forecast accuracy; risk of large errors",
                recommended_action="Investigate root cause; consider ensemble for stability; increase monitoring frequency",
                evidence=[
                    f"{len(unstable_models)} unstable combinations detected",
                    "Coefficient of variation > 30%",
                    "Prediction reliability concern"
                ]
            ))
        
        return patterns
    
    @staticmethod
    def _detect_metric_disagreement(rankings: Dict) -> List[Pattern]:
        """Detect when metrics disagree on best model"""
        patterns = []
        
        winners = set(ranked[0][0] for ranked in rankings.values())
        
        if len(winners) == len(rankings):  # Every metric has different winner
            patterns.append(Pattern(
                pattern_type="METRIC_DISAGREEMENT",
                severity="MODERATE",
                description="Each metric favors a different model",
                business_impact="No universal winner; model performance is context-dependent",
                recommended_action="Define primary success metric or consider ensemble approach",
                evidence=[
                    f"{len(winners)} different winners across {len(rankings)} metrics",
                    "Models excel in different dimensions",
                    "Business priorities will determine choice"
                ]
            ))
        
        return patterns
    
    @staticmethod
    def _detect_ensemble_potential(rankings: Dict, metrics_summary: Dict) -> List[Pattern]:
        """Detect if ensemble approach would be beneficial"""
        patterns = []
        
        # Check diversity
        winners = set(ranked[0][0] for ranked in rankings.values())
        
        # Check if models have complementary strengths
        if len(winners) >= 3:
            patterns.append(Pattern(
                pattern_type="ENSEMBLE_OPPORTUNITY",
                severity="LOW",
                description=f"{len(winners)} models show complementary strengths across metrics",
                business_impact="Ensemble could combine best of each model",
                recommended_action="Prototype ensemble (stacking) to capture complementary strengths",
                evidence=[
                    f"{len(winners)} models excel at different metrics",
                    "Diverse strengths suggest ensemble benefit",
                    "Potential 5-15% performance improvement"
                ]
            ))
        
        return patterns