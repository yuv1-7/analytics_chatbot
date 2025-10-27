from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from typing import List, Optional, Dict, Any
from core.models import EnsembleMember, Model, ModelTypeEnum

class EnsembleRepository:
    """Repository for EnsembleMember and ensemble-related operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_ensemble_composition(self, ensemble_id: str) -> List[EnsembleMember]:
        """Get all base models that make up an ensemble"""
        return self.session.query(EnsembleMember).filter(
            EnsembleMember.ensemble_id == ensemble_id
        ).all()
    
    def get_ensemble_with_models(self, ensemble_id: str) -> Optional[Dict[str, Any]]:
        """Get ensemble details with base model information"""
        ensemble = self.session.query(Model).filter(
            Model.model_id == ensemble_id,
            Model.model_type == ModelTypeEnum.ensemble
        ).first()
        
        if not ensemble:
            return None
        
        members = self.session.query(
            EnsembleMember,
            Model
        ).join(
            Model, 
            EnsembleMember.base_model_id == Model.model_id
        ).filter(
            EnsembleMember.ensemble_id == ensemble_id
        ).all()
        
        base_models = []
        for member, model in members:
            base_models.append({
                'model_id': model.model_id,
                'model_name': model.model_name,
                'algorithm': model.algorithm,
                'weight': member.weight,
                'role': member.role,
                'ensemble_type': member.ensemble_type,
                'configuration': member.configuration
            })
        
        return {
            'ensemble_id': ensemble.model_id,
            'ensemble_name': ensemble.model_name,
            'use_case': ensemble.use_case,
            'version': ensemble.version,
            'algorithm': ensemble.algorithm,
            'description': ensemble.description,
            'base_models': base_models,
            'total_base_models': len(base_models)
        }
    
    def get_base_model_ensembles(self, base_model_id: str) -> List[Dict[str, Any]]:
        """Get all ensembles that include a specific base model"""
        results = self.session.query(
            EnsembleMember,
            Model
        ).join(
            Model,
            EnsembleMember.ensemble_id == Model.model_id
        ).filter(
            EnsembleMember.base_model_id == base_model_id,
            Model.is_active == True
        ).all()
        
        ensembles = []
        for member, ensemble in results:
            ensembles.append({
                'ensemble_id': ensemble.model_id,
                'ensemble_name': ensemble.model_name,
                'use_case': ensemble.use_case,
                'weight': member.weight,
                'role': member.role,
                'ensemble_type': member.ensemble_type
            })
        
        return ensembles
    
    def get_ensembles_by_type(self, ensemble_type: str, use_case: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ensembles by ensemble type (boosting, stacking, etc.)"""
        query = self.session.query(
            EnsembleMember.ensemble_id,
            Model.model_name,
            Model.use_case,
            func.count(EnsembleMember.base_model_id).label('base_model_count')
        ).join(
            Model,
            EnsembleMember.ensemble_id == Model.model_id
        ).filter(
            EnsembleMember.ensemble_type == ensemble_type,
            Model.is_active == True
        )
        
        if use_case:
            query = query.filter(Model.use_case == use_case)
        
        results = query.group_by(
            EnsembleMember.ensemble_id,
            Model.model_name,
            Model.use_case
        ).all()
        
        return [
            {
                'ensemble_id': ensemble_id,
                'ensemble_name': model_name,
                'use_case': use_case_val,
                'base_model_count': count
            }
            for ensemble_id, model_name, use_case_val, count in results
        ]
    
    def get_ensemble_statistics(self, ensemble_id: str) -> Dict[str, Any]:
        """Get statistics about ensemble composition"""
        members = self.get_ensemble_composition(ensemble_id)
        
        if not members:
            return {}
        
        total_weight = sum(m.weight for m in members if m.weight)
        
        roles = {}
        ensemble_types = {}
        algorithms = []
        
        for member in members:
            if member.role:
                roles[member.role] = roles.get(member.role, 0) + 1
            if member.ensemble_type:
                ensemble_types[member.ensemble_type] = ensemble_types.get(member.ensemble_type, 0) + 1
            
            base_model = self.session.query(Model).filter(
                Model.model_id == member.base_model_id
            ).first()
            if base_model:
                algorithms.append(base_model.algorithm)
        
        return {
            'total_base_models': len(members),
            'total_weight': total_weight,
            'roles_distribution': roles,
            'ensemble_types': ensemble_types,
            'algorithms_used': list(set(algorithms)),
            'weighted_ensemble': total_weight > 0
        }
    
    def compare_ensemble_compositions(self, ensemble_id_1: str, ensemble_id_2: str) -> Dict[str, Any]:
        """Compare two ensemble compositions"""
        comp1 = self.get_ensemble_with_models(ensemble_id_1)
        comp2 = self.get_ensemble_with_models(ensemble_id_2)
        
        if not comp1 or not comp2:
            return {}
        
        base_models_1 = {bm['model_id'] for bm in comp1['base_models']}
        base_models_2 = {bm['model_id'] for bm in comp2['base_models']}
        
        common_models = base_models_1.intersection(base_models_2)
        unique_to_1 = base_models_1 - base_models_2
        unique_to_2 = base_models_2 - base_models_1
        
        return {
            'ensemble_1': {
                'id': comp1['ensemble_id'],
                'name': comp1['ensemble_name'],
                'base_model_count': len(base_models_1)
            },
            'ensemble_2': {
                'id': comp2['ensemble_id'],
                'name': comp2['ensemble_name'],
                'base_model_count': len(base_models_2)
            },
            'common_base_models': len(common_models),
            'unique_to_ensemble_1': len(unique_to_1),
            'unique_to_ensemble_2': len(unique_to_2),
            'similarity_score': len(common_models) / len(base_models_1.union(base_models_2)) if base_models_1.union(base_models_2) else 0
        }