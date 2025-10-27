from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from core.models import ModelExecution, Model, ExecutionStatusEnum

class ExecutionRepository:
    """Repository for ModelExecution table operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_id(self, execution_id: str) -> Optional[ModelExecution]:
        """Get execution by ID"""
        return self.session.query(ModelExecution).filter(
            ModelExecution.execution_id == execution_id
        ).first()
    
    def get_latest_execution(self, model_id: str, status: Optional[str] = None) -> Optional[ModelExecution]:
        """Get latest execution for a model"""
        query = self.session.query(ModelExecution).filter(
            ModelExecution.model_id == model_id
        )
        
        if status:
            query = query.filter(ModelExecution.execution_status == status)
        
        return query.order_by(desc(ModelExecution.execution_timestamp)).first()
    
    def get_latest_successful_execution(self, model_id: str) -> Optional[ModelExecution]:
        """Get latest successful execution for a model"""
        return self.get_latest_execution(model_id, ExecutionStatusEnum.success.value)
    
    def get_executions_by_model(
        self, 
        model_id: str, 
        limit: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[ModelExecution]:
        """Get all executions for a model"""
        query = self.session.query(ModelExecution).filter(
            ModelExecution.model_id == model_id
        )
        
        if status:
            query = query.filter(ModelExecution.execution_status == status)
        
        query = query.order_by(desc(ModelExecution.execution_timestamp))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_executions_by_date_range(
        self,
        model_id: str,
        start_date: date,
        end_date: date
    ) -> List[ModelExecution]:
        """Get executions within a date range"""
        return self.session.query(ModelExecution).filter(
            ModelExecution.model_id == model_id,
            ModelExecution.prediction_date >= start_date,
            ModelExecution.prediction_date <= end_date,
            ModelExecution.execution_status == ExecutionStatusEnum.success
        ).order_by(desc(ModelExecution.execution_timestamp)).all()
    
    def get_executions_with_drift(self, model_id: Optional[str] = None) -> List[ModelExecution]:
        """Get executions where drift was detected"""
        query = self.session.query(ModelExecution).filter(
            ModelExecution.drift_detected == True
        )
        
        if model_id:
            query = query.filter(ModelExecution.model_id == model_id)
        
        return query.order_by(desc(ModelExecution.execution_timestamp)).all()
    
    def get_failed_executions(
        self,
        model_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ModelExecution]:
        """Get recent failed executions"""
        query = self.session.query(ModelExecution).filter(
            ModelExecution.execution_status == ExecutionStatusEnum.failed
        )
        
        if model_id:
            query = query.filter(ModelExecution.model_id == model_id)
        
        return query.order_by(desc(ModelExecution.execution_timestamp)).limit(limit).all()
    
    def get_executions_by_prediction_date(
        self,
        prediction_date: date,
        use_case: Optional[str] = None
    ) -> List[ModelExecution]:
        """Get all executions for a specific prediction date"""
        query = self.session.query(ModelExecution).join(Model).filter(
            ModelExecution.prediction_date == prediction_date,
            ModelExecution.execution_status == ExecutionStatusEnum.success
        )
        
        if use_case:
            query = query.filter(Model.use_case == use_case)
        
        return query.order_by(desc(ModelExecution.execution_timestamp)).all()
    
    def count_executions_by_status(self, model_id: Optional[str] = None) -> Dict[str, int]:
        """Count executions by status"""
        query = self.session.query(
            ModelExecution.execution_status,
            func.count(ModelExecution.execution_id)
        )
        
        if model_id:
            query = query.filter(ModelExecution.model_id == model_id)
        
        results = query.group_by(ModelExecution.execution_status).all()
        
        return {str(status.value): count for status, count in results}
    
    def get_execution_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get execution statistics for a model"""
        executions = self.get_executions_by_model(model_id)
        
        if not executions:
            return {}
        
        successful = [e for e in executions if e.execution_status == ExecutionStatusEnum.success]
        failed = [e for e in executions if e.execution_status == ExecutionStatusEnum.failed]
        with_drift = [e for e in executions if e.drift_detected]
        
        avg_runtime = None
        if successful:
            runtimes = [e.runtime_seconds for e in successful if e.runtime_seconds]
            if runtimes:
                avg_runtime = sum(runtimes) / len(runtimes)
        
        return {
            'total_executions': len(executions),
            'successful_executions': len(successful),
            'failed_executions': len(failed),
            'drift_detected_count': len(with_drift),
            'success_rate': len(successful) / len(executions) if executions else 0,
            'drift_rate': len(with_drift) / len(successful) if successful else 0,
            'average_runtime_seconds': avg_runtime,
            'latest_execution': executions[0].execution_timestamp if executions else None
        }
    
    def get_baseline_execution(self, execution_id: str) -> Optional[ModelExecution]:
        """Get the baseline execution for comparison"""
        execution = self.get_by_id(execution_id)
        if execution and execution.baseline_execution_id:
            return self.get_by_id(execution.baseline_execution_id)
        return None