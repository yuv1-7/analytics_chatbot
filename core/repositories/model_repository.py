from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime
from core.models import Model, ModelExecution, EnsembleMember, ModelTypeEnum

class ModelRepository:
    """Repository for Model table operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_id(self, model_id: str) -> Optional[Model]:
        """Get model by ID"""
        return self.session.query(Model).filter(Model.model_id == model_id).first()
    
    def get_by_name(self, model_name: str) -> Optional[Model]:
        """Get model by name (latest version)"""
        return self.session.query(Model).filter(
            Model.model_name == model_name,
            Model.is_active == True
        ).order_by(desc(Model.created_at)).first()
    
    def get_by_name_and_version(self, model_name: str, version: str) -> Optional[Model]:
        """Get specific model version"""
        return self.session.query(Model).filter(
            Model.model_name == model_name,
            Model.version == version
        ).first()
    
    def get_active_models(self, model_type: Optional[str] = None, use_case: Optional[str] = None) -> List[Model]:
        """Get all active models, optionally filtered"""
        query = self.session.query(Model).filter(Model.is_active == True)
        
        if model_type:
            query = query.filter(Model.model_type == model_type)
        
        if use_case:
            query = query.filter(Model.use_case == use_case)
        
        return query.order_by(desc(Model.created_at)).all()
    
    def get_models_by_use_case(self, use_case: str) -> List[Model]:
        """Get all models for a specific use case"""
        return self.session.query(Model).filter(
            Model.use_case == use_case,
            Model.is_active == True
        ).order_by(desc(Model.created_at)).all()
    
    def get_ensemble_models(self, use_case: Optional[str] = None) -> List[Model]:
        """Get all ensemble models"""
        query = self.session.query(Model).filter(
            Model.model_type == ModelTypeEnum.ensemble,
            Model.is_active == True
        )
        
        if use_case:
            query = query.filter(Model.use_case == use_case)
        
        return query.order_by(desc(Model.created_at)).all()
    
    def get_base_models(self, use_case: Optional[str] = None) -> List[Model]:
        """Get all base models"""
        query = self.session.query(Model).filter(
            Model.model_type == ModelTypeEnum.base_model,
            Model.is_active == True
        )
        
        if use_case:
            query = query.filter(Model.use_case == use_case)
        
        return query.order_by(desc(Model.created_at)).all()
    
    def search_models(self, search_term: str) -> List[Model]:
        """Search models by name or description"""
        search = f"%{search_term}%"
        return self.session.query(Model).filter(
            or_(
                Model.model_name.ilike(search),
                Model.description.ilike(search),
                Model.algorithm.ilike(search)
            ),
            Model.is_active == True
        ).all()
    
    def get_model_versions(self, model_name: str) -> List[Model]:
        """Get all versions of a model"""
        return self.session.query(Model).filter(
            Model.model_name == model_name
        ).order_by(desc(Model.created_at)).all()
    
    def get_models_by_algorithm(self, algorithm: str) -> List[Model]:
        """Get models by algorithm type"""
        return self.session.query(Model).filter(
            Model.algorithm.ilike(f"%{algorithm}%"),
            Model.is_active == True
        ).all()
    
    def count_models_by_type(self) -> Dict[str, int]:
        """Count models by type"""
        results = self.session.query(
            Model.model_type,
            func.count(Model.model_id)
        ).filter(Model.is_active == True).group_by(Model.model_type).all()
        
        return {str(model_type.value): count for model_type, count in results}
    
    def count_models_by_use_case(self) -> Dict[str, int]:
        """Count models by use case"""
        results = self.session.query(
            Model.use_case,
            func.count(Model.model_id)
        ).filter(Model.is_active == True).group_by(Model.use_case).all()
        
        return {use_case: count for use_case, count in results}