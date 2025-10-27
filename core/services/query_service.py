from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta
from core.database import get_db_session
from core.repositories.model_repository import ModelRepository
from core.repositories.execution_repository import ExecutionRepository
from core.repositories.ensemble_repository import EnsembleRepository
from core.repositories.metrics_repository import MetricsRepository
from core.repositories.prediction_repository import PredictionRepository

class QueryService:
    """
    High-level service for handling complex queries across multiple repositories.
    This is the main interface for the agent to retrieve data.
    """
    
    @staticmethod
    def get_ensemble_vs_base_performance(
        ensemble_name: str,
        use_case: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare ensemble performance against its base models.
        
        Args:
            ensemble_name: Name of the ensemble model
            use_case: Optional use case filter
            metrics: Optional list of specific metrics to compare
        
        Returns:
            Dict with comparison results and analysis
        """
        with get_db_session() as session:
            model_repo = ModelRepository(session)
            metrics_repo = MetricsRepository(session)
            ensemble_repo = EnsembleRepository(session)
            
            # Find ensemble model
            ensemble = model_repo.get_by_name(ensemble_name)
            if not ensemble:
                return {'error': f'Ensemble model "{ensemble_name}" not found'}
            
            if ensemble.model_type.value != 'ensemble':
                return {'error': f'Model "{ensemble_name}" is not an ensemble'}
            
            # Get composition
            composition = ensemble_repo.get_ensemble_with_models(ensemble.model_id)
            
            # Get performance comparison
            comparison = metrics_repo.compare_ensemble_vs_base_models(
                ensemble.model_id,
                metric_names=metrics
            )
            
            return {
                **comparison,
                'composition': composition
            }
    
    @staticmethod
    def get_model_performance_summary(
        model_name: Optional[str] = None,
        use_case: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary for models.
        
        Args:
            model_name: Specific model name
            use_case: Filter by use case
            model_type: Filter by model type (base_model/ensemble)
        
        Returns:
            Dict with model performance data
        """
        with get_db_session() as session:
            model_repo = ModelRepository(session)
            execution_repo = ExecutionRepository(session)
            metrics_repo = MetricsRepository(session)
            
            # Get models
            if model_name:
                models = [model_repo.get_by_name(model_name)]
                models = [m for m in models if m]  # Filter None
            else:
                models = model_repo.get_active_models(model_type, use_case)
            
            if not models:
                return {'error': 'No models found matching criteria'}
            
            results = []
            for model in models:
                latest_exec = execution_repo.get_latest_successful_execution(model.model_id)
                if not latest_exec:
                    continue
                
                metrics = metrics_repo.get_metrics_by_execution(latest_exec.execution_id, 'test')
                exec_stats = execution_repo.get_execution_statistics(model.model_id)
                
                results.append({
                    'model_id': model.model_id,
                    'model_name': model.model_name,
                    'model_type': model.model_type.value,
                    'use_case': model.use_case,
                    'version': model.version,
                    'algorithm': model.algorithm,
                    'latest_execution': {
                        'execution_id': latest_exec.execution_id,
                        'timestamp': latest_exec.execution_timestamp,
                        'prediction_date': latest_exec.prediction_date
                    },
                    'metrics': {m.metric_name: m.metric_value for m in metrics},
                    'execution_statistics': exec_stats
                })
            
            return {
                'total_models': len(results),
                'models': results
            }
    
    @staticmethod
    def get_drift_detection_summary(
        model_name: Optional[str] = None,
        use_case: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get drift detection summary for models.
        
        Args:
            model_name: Specific model name
            use_case: Filter by use case
            days_back: Number of days to look back
        
        Returns:
            Dict with drift detection results
        """
        with get_db_session() as session:
            model_repo = ModelRepository(session)
            execution_repo = ExecutionRepository(session)
            metrics_repo = MetricsRepository(session)
            
            # Get models
            if model_name:
                models = [model_repo.get_by_name(model_name)]
                models = [m for m in models if m]
            else:
                models = model_repo.get_active_models(use_case=use_case)
            
            if not models:
                return {'error': 'No models found'}
            
            results = []
            for model in models:
                drift_history = metrics_repo.get_drift_history(model.model_id, limit=10)
                latest_exec = execution_repo.get_latest_execution(model.model_id)
                
                drift_results = None
                if latest_exec and latest_exec.drift_detected:
                    drift_results = metrics_repo.get_drift_results(
                        latest_exec.execution_id,
                        significant_only=True
                    )
                
                results.append({
                    'model_name': model.model_name,
                    'model_id': model.model_id,
                    'use_case': model.use_case,
                    'current_drift_detected': latest_exec.drift_detected if latest_exec else False,
                    'current_drift_score': latest_exec.drift_score if latest_exec else None,
                    'drift_history': drift_history,
                    'recent_drift_details': [
                        {
                            'drift_type': d.drift_type.value if d.drift_type else None,
                            'drift_metric': d.drift_metric,
                            'drift_score': d.drift_score,
                            'explanation': d.drift_explanation
                        }
                        for d in (drift_results or [])
                    ]
                })
            
            return {
                'total_models_checked': len(results),
                'models_with_drift': len([r for r in results if r['current_drift_detected']]),
                'drift_details': results
            }
    
    @staticmethod
    def compare_model_versions(
        model_name: str,
        old_version: str,
        new_version: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two versions of the same model.
        
        Args:
            model_name: Name of the model
            old_version: Version string for old version
            new_version: Version string for new version
            metrics: Optional list of metrics to compare
        
        Returns:
            Dict with version comparison
        """
        with get_db_session() as session:
            model_repo = ModelRepository(session)
            execution_repo = ExecutionRepository(session)
            metrics_repo = MetricsRepository(session)
            
            # Get both versions
            old_model = model_repo.get_by_name_and_version(model_name, old_version)
            new_model = model_repo.get_by_name_and_version(model_name, new_version)
            
            if not old_model:
                return {'error': f'Version {old_version} not found'}
            if not new_model:
                return {'error': f'Version {new_version} not found'}
            
            # Get latest executions
            old_exec = execution_repo.get_latest_successful_execution(old_model.model_id)
            new_exec = execution_repo.get_latest_successful_execution(new_model.model_id)
            
            if not old_exec or not new_exec:
                return {'error': 'Missing execution data for one or both versions'}
            
            # Compare metrics
            comparison = metrics_repo.get_metrics_comparison(
                old_exec.execution_id,
                new_exec.execution_id
            )
            
            # Compare feature importance
            feature_comparison = metrics_repo.compare_feature_importance(
                old_exec.execution_id,
                new_exec.execution_id,
                top_n=15
            )
            
            return {
                'model_name': model_name,
                'old_version': {
                    'version': old_version,
                    'execution_id': old_exec.execution_id,
                    'execution_timestamp': old_exec.execution_timestamp
                },
                'new_version': {
                    'version': new_version,
                    'execution_id': new_exec.execution_id,
                    'execution_timestamp': new_exec.execution_timestamp
                },
                'metrics_comparison': comparison,
                'feature_importance_comparison': feature_comparison
            }
    
    @staticmethod
    def get_feature_importance_analysis(
        model_name: str,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Get detailed feature importance analysis for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
        
        Returns:
            Dict with feature importance data
        """
        with get_db_session() as session:
            model_repo = ModelRepository(session)
            execution_repo = ExecutionRepository(session)
            metrics_repo = MetricsRepository(session)
            
            model = model_repo.get_by_name(model_name)
            if not model:
                return {'error': f'Model "{model_name}" not found'}
            
            latest_exec = execution_repo.get_latest_successful_execution(model.model_id)
            if not latest_exec:
                return {'error': 'No successful execution found'}
            
            features = metrics_repo.get_feature_importance(
                latest_exec.execution_id,
                top_n=top_n
            )
            
            return {
                'model_name': model.model_name,
                'model_id': model.model_id,
                'execution_id': latest_exec.execution_id,
                'execution_timestamp': latest_exec.execution_timestamp,
                'top_features': [
                    {
                        'rank': f.rank,
                        'feature_name': f.feature_name,
                        'importance_score': f.importance_score,
                        'importance_type': f.importance_type
                    }
                    for f in features
                ]
            }
    
    @staticmethod
    def get_prediction_analysis(
        model_name: str,
        entity_type: Optional[str] = None,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Get prediction analysis for a model.
        
        Args:
            model_name: Name of the model
            entity_type: Optional entity type filter (HCP, territory, etc.)
            top_n: Number of top predictions to return
        
        Returns:
            Dict with prediction analysis
        """
        with get_db_session() as session:
            model_repo = ModelRepository(session)
            execution_repo = ExecutionRepository(session)
            prediction_repo = PredictionRepository(session)
            
            model = model_repo.get_by_name(model_name)
            if not model:
                return {'error': f'Model "{model_name}" not found'}
            
            latest_exec = execution_repo.get_latest_successful_execution(model.model_id)
            if not latest_exec:
                return {'error': 'No successful execution found'}
            
            # Get summary
            summary = prediction_repo.get_predictions_summary(latest_exec.execution_id)
            
            # Get top predictions
            top_predictions = prediction_repo.get_top_predictions(
                latest_exec.execution_id,
                top_n=top_n
            )
            
            # Calculate accuracy if actuals available
            accuracy = prediction_repo.calculate_prediction_accuracy(latest_exec.execution_id)
            
            return {
                'model_name': model.model_name,
                'execution_id': latest_exec.execution_id,
                'execution_timestamp': latest_exec.execution_timestamp,
                'summary': summary,
                'accuracy_metrics': accuracy,
                'top_predictions': [
                    {
                        'entity_type': p.entity_type.value if hasattr(p.entity_type, 'value') else p.entity_type,
                        'entity_id': p.entity_id,
                        'prediction_value': p.prediction_value,
                        'prediction_probability': p.prediction_probability,
                        'actual_value': p.actual_value,
                        'residual': p.residual
                    }
                    for p in top_predictions
                ]
            }
    
    @staticmethod
    def search_models(search_term: str) -> Dict[str, Any]:
        """
        Search for models by name, description, or algorithm.
        
        Args:
            search_term: Search term
        
        Returns:
            Dict with search results
        """
        with get_db_session() as session:
            model_repo = ModelRepository(session)
            
            models = model_repo.search_models(search_term)
            
            return {
                'search_term': search_term,
                'total_results': len(models),
                'models': [
                    {
                        'model_id': m.model_id,
                        'model_name': m.model_name,
                        'model_type': m.model_type.value,
                        'use_case': m.use_case,
                        'algorithm': m.algorithm,
                        'description': m.description,
                        'version': m.version
                    }
                    for m in models
                ]
            }