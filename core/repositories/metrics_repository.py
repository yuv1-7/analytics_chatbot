from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
from typing import List, Optional, Dict, Any
from core.models import (
    PerformanceMetric, ModelExecution, Model, 
    FeatureImportance, DriftDetectionResult,
    DataSplitEnum, ModelTypeEnum
)
import pandas as pd

class MetricsRepository:
    """Repository for performance metrics and feature importance"""
    
    def __init__(self, session: Session):
        self.session = session
    
    # ============= Performance Metrics =============
    
    def get_metrics_by_execution(
        self, 
        execution_id: str,
        data_split: Optional[str] = None
    ) -> List[PerformanceMetric]:
        """Get all metrics for an execution"""
        query = self.session.query(PerformanceMetric).filter(
            PerformanceMetric.execution_id == execution_id
        )
        
        if data_split:
            query = query.filter(PerformanceMetric.data_split == data_split)
        
        return query.all()
    
    def get_metric_by_name(
        self,
        execution_id: str,
        metric_name: str,
        data_split: str = 'test'
    ) -> Optional[PerformanceMetric]:
        """Get specific metric for an execution"""
        return self.session.query(PerformanceMetric).filter(
            PerformanceMetric.execution_id == execution_id,
            PerformanceMetric.metric_name == metric_name,
            PerformanceMetric.data_split == data_split
        ).first()
    
    def get_metrics_comparison(
        self,
        execution_id_1: str,
        execution_id_2: str,
        data_split: str = 'test'
    ) -> Dict[str, Any]:
        """Compare metrics between two executions"""
        metrics_1 = self.get_metrics_by_execution(execution_id_1, data_split)
        metrics_2 = self.get_metrics_by_execution(execution_id_2, data_split)
        
        metrics_dict_1 = {m.metric_name: m.metric_value for m in metrics_1}
        metrics_dict_2 = {m.metric_name: m.metric_value for m in metrics_2}
        
        comparison = {}
        for metric_name in set(metrics_dict_1.keys()).union(metrics_dict_2.keys()):
            val_1 = metrics_dict_1.get(metric_name)
            val_2 = metrics_dict_2.get(metric_name)
            
            if val_1 is not None and val_2 is not None:
                diff = val_2 - val_1
                pct_change = (diff / val_1 * 100) if val_1 != 0 else None
                
                comparison[metric_name] = {
                    'execution_1_value': val_1,
                    'execution_2_value': val_2,
                    'absolute_difference': diff,
                    'percent_change': pct_change
                }
        
        return comparison
    
    def get_metric_trend(
        self,
        model_id: str,
        metric_name: str,
        data_split: str = 'test',
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get metric values over time for a model"""
        results = self.session.query(
            ModelExecution.execution_timestamp,
            ModelExecution.prediction_date,
            PerformanceMetric.metric_value
        ).join(
            PerformanceMetric,
            ModelExecution.execution_id == PerformanceMetric.execution_id
        ).filter(
            ModelExecution.model_id == model_id,
            PerformanceMetric.metric_name == metric_name,
            PerformanceMetric.data_split == data_split
        ).order_by(desc(ModelExecution.execution_timestamp)).limit(limit).all()
        
        return [
            {
                'execution_timestamp': timestamp,
                'prediction_date': pred_date,
                'metric_value': value
            }
            for timestamp, pred_date, value in results
        ]
    
    def get_best_execution_by_metric(
        self,
        model_id: str,
        metric_name: str,
        data_split: str = 'test',
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Find best execution based on a metric"""
        order_func = desc if maximize else func.asc
        
        result = self.session.query(
            ModelExecution,
            PerformanceMetric
        ).join(
            PerformanceMetric,
            ModelExecution.execution_id == PerformanceMetric.execution_id
        ).filter(
            ModelExecution.model_id == model_id,
            PerformanceMetric.metric_name == metric_name,
            PerformanceMetric.data_split == data_split
        ).order_by(order_func(PerformanceMetric.metric_value)).first()
        
        if result:
            execution, metric = result
            return {
                'execution_id': execution.execution_id,
                'execution_timestamp': execution.execution_timestamp,
                'metric_value': metric.metric_value
            }
        return None
    
    # ============= Ensemble vs Base Comparison =============
    
    def compare_ensemble_vs_base_models(
        self,
        ensemble_id: str,
        metric_names: Optional[List[str]] = None,
        data_split: str = 'test'
    ) -> Dict[str, Any]:
        """Compare ensemble performance against its base models"""
        from core.repositories.ensemble_repository import EnsembleRepository
        from core.repositories.execution_repository import ExecutionRepository
        
        ensemble_repo = EnsembleRepository(self.session)
        execution_repo = ExecutionRepository(self.session)
        
        # Get ensemble composition
        composition = ensemble_repo.get_ensemble_with_models(ensemble_id)
        if not composition:
            return {}
        
        # Get latest execution for ensemble
        ensemble_execution = execution_repo.get_latest_successful_execution(ensemble_id)
        if not ensemble_execution:
            return {'error': 'No successful execution found for ensemble'}
        
        # Get ensemble metrics
        ensemble_metrics = self.get_metrics_by_execution(ensemble_execution.execution_id, data_split)
        
        # Get base model metrics
        base_models_metrics = []
        for base_model in composition['base_models']:
            base_execution = execution_repo.get_latest_successful_execution(base_model['model_id'])
            if base_execution:
                base_metrics = self.get_metrics_by_execution(base_execution.execution_id, data_split)
                base_models_metrics.append({
                    'model_id': base_model['model_id'],
                    'model_name': base_model['model_name'],
                    'algorithm': base_model['algorithm'],
                    'metrics': {m.metric_name: m.metric_value for m in base_metrics}
                })
        
        # Calculate comparison
        ensemble_metrics_dict = {m.metric_name: m.metric_value for m in ensemble_metrics}
        
        if metric_names:
            metrics_to_compare = metric_names
        else:
            metrics_to_compare = list(ensemble_metrics_dict.keys())
        
        comparison_results = {}
        for metric_name in metrics_to_compare:
            if metric_name not in ensemble_metrics_dict:
                continue
            
            ensemble_value = ensemble_metrics_dict[metric_name]
            base_values = [
                bm['metrics'].get(metric_name) 
                for bm in base_models_metrics 
                if metric_name in bm['metrics']
            ]
            
            if not base_values:
                continue
            
            avg_base = sum(base_values) / len(base_values)
            best_base = max(base_values)
            worst_base = min(base_values)
            
            # Determine if higher is better (this is simplified, should be metric-specific)
            lower_is_better = metric_name.lower() in ['rmse', 'mae', 'mse', 'error']
            
            if lower_is_better:
                improvement_vs_avg = ((avg_base - ensemble_value) / avg_base * 100) if avg_base != 0 else 0
                improvement_vs_best = ((best_base - ensemble_value) / best_base * 100) if best_base != 0 else 0
            else:
                improvement_vs_avg = ((ensemble_value - avg_base) / avg_base * 100) if avg_base != 0 else 0
                improvement_vs_best = ((ensemble_value - best_base) / best_base * 100) if best_base != 0 else 0
            
            comparison_results[metric_name] = {
                'ensemble_value': ensemble_value,
                'base_average': avg_base,
                'base_best': best_base,
                'base_worst': worst_base,
                'improvement_vs_average': improvement_vs_avg,
                'improvement_vs_best': improvement_vs_best,
                'individual_base_values': base_values
            }
        
        return {
            'ensemble_name': composition['ensemble_name'],
            'ensemble_id': ensemble_id,
            'base_model_count': len(base_models_metrics),
            'comparison': comparison_results,
            'base_models': base_models_metrics
        }
    
    # ============= Feature Importance =============
    
    def get_feature_importance(
        self,
        execution_id: str,
        top_n: Optional[int] = None,
        importance_type: Optional[str] = None
    ) -> List[FeatureImportance]:
        """Get feature importance for an execution"""
        query = self.session.query(FeatureImportance).filter(
            FeatureImportance.execution_id == execution_id
        )
        
        if importance_type:
            query = query.filter(FeatureImportance.importance_type == importance_type)
        
        query = query.order_by(FeatureImportance.rank)
        
        if top_n:
            query = query.limit(top_n)
        
        return query.all()
    
    def compare_feature_importance(
        self,
        execution_id_1: str,
        execution_id_2: str,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Compare feature importance between two executions"""
        features_1 = self.get_feature_importance(execution_id_1, top_n)
        features_2 = self.get_feature_importance(execution_id_2, top_n)
        
        features_dict_1 = {f.feature_name: f.importance_score for f in features_1}
        features_dict_2 = {f.feature_name: f.importance_score for f in features_2}
        
        all_features = set(features_dict_1.keys()).union(features_dict_2.keys())
        
        comparison = []
        for feature in all_features:
            score_1 = features_dict_1.get(feature)
            score_2 = features_dict_2.get(feature)
            
            comparison.append({
                'feature_name': feature,
                'execution_1_score': score_1,
                'execution_2_score': score_2,
                'score_difference': (score_2 - score_1) if (score_1 and score_2) else None,
                'in_both': (score_1 is not None and score_2 is not None)
            })
        
        # Sort by absolute difference
        comparison.sort(
            key=lambda x: abs(x['score_difference']) if x['score_difference'] else 0,
            reverse=True
        )
        
        return {
            'features_in_common': len([c for c in comparison if c['in_both']]),
            'features_unique_to_execution_1': len([c for c in comparison if c['execution_1_score'] and not c['execution_2_score']]),
            'features_unique_to_execution_2': len([c for c in comparison if c['execution_2_score'] and not c['execution_1_score']]),
            'comparison': comparison
        }
    
    # ============= Drift Detection =============
    
    def get_drift_results(
        self,
        execution_id: str,
        significant_only: bool = False
    ) -> List[DriftDetectionResult]:
        """Get drift detection results for an execution"""
        query = self.session.query(DriftDetectionResult).filter(
            DriftDetectionResult.execution_id == execution_id
        )
        
        if significant_only:
            query = query.filter(DriftDetectionResult.is_significant == True)
        
        return query.order_by(desc(DriftDetectionResult.drift_score)).all()
    
    def get_drift_history(
        self,
        model_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get drift detection history for a model"""
        results = self.session.query(
            ModelExecution.execution_timestamp,
            ModelExecution.prediction_date,
            ModelExecution.drift_detected,
            ModelExecution.drift_score
        ).filter(
            ModelExecution.model_id == model_id,
            ModelExecution.drift_detected == True
        ).order_by(desc(ModelExecution.execution_timestamp)).limit(limit).all()
        
        return [
            {
                'execution_timestamp': timestamp,
                'prediction_date': pred_date,
                'drift_detected': drift_detected,
                'drift_score': drift_score
            }
            for timestamp, pred_date, drift_detected, drift_score in results
        ]
    
    def get_metrics_as_dataframe(
        self,
        execution_ids: List[str],
        data_split: str = 'test'
    ) -> pd.DataFrame:
        """Get metrics as pandas DataFrame for analysis"""
        results = self.session.query(
            PerformanceMetric.execution_id,
            PerformanceMetric.metric_name,
            PerformanceMetric.metric_value
        ).filter(
            PerformanceMetric.execution_id.in_(execution_ids),
            PerformanceMetric.data_split == data_split
        ).all()
        
        data = [
            {
                'execution_id': exec_id,
                'metric_name': metric_name,
                'metric_value': metric_value
            }
            for exec_id, metric_name, metric_value in results
        ]
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.pivot(index='execution_id', columns='metric_name', values='metric_value')
        
        return df