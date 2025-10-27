from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
from typing import List, Optional, Dict, Any
from datetime import date
from core.models import Prediction, ModelExecution, EntityTypeEnum
import pandas as pd

class PredictionRepository:
    """Repository for Prediction table operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_predictions_by_execution(
        self,
        execution_id: str,
        limit: Optional[int] = None
    ) -> List[Prediction]:
        """Get all predictions for an execution"""
        query = self.session.query(Prediction).filter(
            Prediction.execution_id == execution_id
        )
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_predictions_by_entity(
        self,
        entity_type: str,
        entity_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Prediction]:
        """Get predictions for a specific entity"""
        query = self.session.query(Prediction).filter(
            Prediction.entity_type == entity_type,
            Prediction.entity_id == entity_id
        )
        
        if start_date:
            query = query.filter(Prediction.prediction_date >= start_date)
        if end_date:
            query = query.filter(Prediction.prediction_date <= end_date)
        
        return query.order_by(desc(Prediction.prediction_date)).all()
    
    def get_predictions_with_actuals(
        self,
        execution_id: str,
        limit: Optional[int] = None
    ) -> List[Prediction]:
        """Get predictions that have actual values for comparison"""
        query = self.session.query(Prediction).filter(
            Prediction.execution_id == execution_id,
            Prediction.actual_value.isnot(None)
        )
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def calculate_prediction_accuracy(self, execution_id: str) -> Dict[str, Any]:
        """Calculate prediction accuracy metrics"""
        predictions = self.get_predictions_with_actuals(execution_id)
        
        if not predictions:
            return {'error': 'No predictions with actual values found'}
        
        residuals = [p.residual for p in predictions if p.residual is not None]
        
        if not residuals:
            return {'error': 'No residuals calculated'}
        
        mae = sum(abs(r) for r in residuals) / len(residuals)
        mse = sum(r**2 for r in residuals) / len(residuals)
        rmse = mse ** 0.5
        
        actual_values = [p.actual_value for p in predictions]
        predicted_values = [p.prediction_value for p in predictions]
        
        mean_actual = sum(actual_values) / len(actual_values)
        ss_tot = sum((a - mean_actual)**2 for a in actual_values)
        ss_res = sum(r**2 for r in residuals)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'total_predictions': len(predictions),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mean_absolute_residual': mae,
            'max_residual': max(residuals),
            'min_residual': min(residuals)
        }
    
    def get_top_predictions(
        self,
        execution_id: str,
        top_n: int = 10,
        by_value: bool = True
    ) -> List[Prediction]:
        """Get top N predictions by value or probability"""
        query = self.session.query(Prediction).filter(
            Prediction.execution_id == execution_id
        )
        
        if by_value:
            query = query.order_by(desc(Prediction.prediction_value))
        else:
            query = query.order_by(desc(Prediction.prediction_probability))
        
        return query.limit(top_n).all()
    
    def get_predictions_by_date(
        self,
        prediction_date: date,
        entity_type: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> List[Prediction]:
        """Get all predictions for a specific date"""
        query = self.session.query(Prediction).join(ModelExecution).filter(
            Prediction.prediction_date == prediction_date
        )
        
        if entity_type:
            query = query.filter(Prediction.entity_type == entity_type)
        
        if model_id:
            query = query.filter(ModelExecution.model_id == model_id)
        
        return query.all()
    
    def compare_predictions(
        self,
        execution_id_1: str,
        execution_id_2: str,
        entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare predictions between two executions"""
        query_1 = self.session.query(Prediction).filter(
            Prediction.execution_id == execution_id_1
        )
        query_2 = self.session.query(Prediction).filter(
            Prediction.execution_id == execution_id_2
        )
        
        if entity_type:
            query_1 = query_1.filter(Prediction.entity_type == entity_type)
            query_2 = query_2.filter(Prediction.entity_type == entity_type)
        
        predictions_1 = {p.entity_id: p for p in query_1.all()}
        predictions_2 = {p.entity_id: p for p in query_2.all()}
        
        common_entities = set(predictions_1.keys()).intersection(predictions_2.keys())
        
        differences = []
        for entity_id in common_entities:
            p1 = predictions_1[entity_id]
            p2 = predictions_2[entity_id]
            
            diff = p2.prediction_value - p1.prediction_value
            pct_change = (diff / p1.prediction_value * 100) if p1.prediction_value != 0 else None
            
            differences.append({
                'entity_id': entity_id,
                'entity_type': p1.entity_type,
                'execution_1_prediction': p1.prediction_value,
                'execution_2_prediction': p2.prediction_value,
                'absolute_difference': diff,
                'percent_change': pct_change
            })
        
        # Sort by absolute difference
        differences.sort(key=lambda x: abs(x['absolute_difference']), reverse=True)
        
        return {
            'total_comparisons': len(differences),
            'entities_only_in_execution_1': len(predictions_1) - len(common_entities),
            'entities_only_in_execution_2': len(predictions_2) - len(common_entities),
            'differences': differences
        }
    
    def get_predictions_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get summary statistics for predictions"""
        predictions = self.get_predictions_by_execution(execution_id)
        
        if not predictions:
            return {'error': 'No predictions found'}
        
        values = [p.prediction_value for p in predictions]
        
        entity_counts = {}
        for p in predictions:
            entity_type = p.entity_type.value if hasattr(p.entity_type, 'value') else p.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        with_actuals = len([p for p in predictions if p.actual_value is not None])
        
        return {
            'total_predictions': len(predictions),
            'predictions_with_actuals': with_actuals,
            'entity_type_distribution': entity_counts,
            'prediction_value_stats': {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'median': sorted(values)[len(values)//2]
            }
        }
    
    def get_predictions_as_dataframe(
        self,
        execution_id: str,
        include_actuals: bool = False
    ) -> pd.DataFrame:
        """Get predictions as pandas DataFrame"""
        if include_actuals:
            predictions = self.get_predictions_with_actuals(execution_id)
        else:
            predictions = self.get_predictions_by_execution(execution_id)
        
        data = []
        for p in predictions:
            row = {
                'entity_type': p.entity_type.value if hasattr(p.entity_type, 'value') else p.entity_type,
                'entity_id': p.entity_id,
                'prediction_value': p.prediction_value,
                'prediction_date': p.prediction_date
            }
            
            if p.actual_value is not None:
                row['actual_value'] = p.actual_value
                row['residual'] = p.residual
            
            if p.prediction_probability is not None:
                row['prediction_probability'] = p.prediction_probability
            
            data.append(row)
        
        return pd.DataFrame(data)