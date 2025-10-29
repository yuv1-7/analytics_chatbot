"""
Vector Database Context Documents for Pharma Commercial Analytics

This module contains structured knowledge documents to be embedded and stored
in a vector database for semantic retrieval during query processing.

Each document is structured with:
- doc_id: Unique identifier
- category: Document category for filtering
- title: Document title
- content: Main content for embedding
- metadata: Additional structured information
- keywords: Key terms for hybrid search
"""

VECTOR_DB_DOCUMENTS = [
    
    # ========================================================================
    # CATEGORY: USE CASES
    # ========================================================================
    
    {
        "doc_id": "UC001",
        "category": "use_case",
        "title": "NRx Forecasting Use Case",
        "content": """
        NRx (New Prescription) Forecasting predicts the number of new prescriptions 
        that Healthcare Providers (HCPs) will write for a pharmaceutical product.
        
        Business Context:
        - NRx is a leading indicator of market adoption and HCP behavior change
        - Different from TRx (Total Prescriptions = New + Refills)
        - Critical for launch planning and resource allocation
        - Typically forecasted monthly or quarterly by HCP, territory, or segment
        
        Typical Models Used:
        - Base models: Random Forest, XGBoost, LightGBM (gradient boosting)
        - Ensemble approach: Stacking with meta-learner combining multiple base models
        - Time series components may be incorporated
        
        Key Features:
        - HCP specialty, prescribing history, patient volume
        - Historical prescription trends (past 6-12 months)
        - Marketing touchpoints (calls, emails, events, samples)
        - Competitive intelligence and market share
        - Geographic and demographic factors
        - Seasonal patterns
        
        Success Metrics:
        - RMSE (Root Mean Squared Error) - lower is better
        - MAE (Mean Absolute Error) - lower is better
        - R² Score - higher is better (0-1 range)
        - MAPE (Mean Absolute Percentage Error) for interpretability
        
        Business Impact:
        - Accurate forecasts drive field force allocation
        - Identify high-potential HCPs for targeting
        - Optimize marketing spend and sample distribution
        - Track launch trajectory vs. plan
        """,
        "metadata": {
            "typical_algorithms": ["Random Forest", "XGBoost", "LightGBM"],
            "ensemble_types": ["stacking", "boosting", "meta_learning"],
            "key_metrics": ["RMSE", "MAE", "R2", "MAPE"],
            "prediction_target": "new_prescriptions",
            "typical_features": ["hcp_specialty", "prescribing_history", "marketing_touches", "patient_volume"]
        },
        "keywords": ["NRx", "new prescriptions", "forecasting", "HCP behavior", "launch planning"]
    },
    
    {
        "doc_id": "UC002",
        "category": "use_case",
        "title": "HCP Response & Engagement Prediction",
        "content": """
        HCP Response Prediction models forecast how Healthcare Providers will respond 
        to marketing messages, campaigns, and promotional activities.
        
        Business Context:
        - Predicts likelihood of HCP engagement (email open, call acceptance, event attendance)
        - Forecasts prescription behavior change following promotional activity
        - Enables personalized messaging and channel optimization
        - Critical for maximizing marketing ROI and field force efficiency
        
        Typical Models Used:
        - Base models: Logistic Regression, SVM, Neural Networks, Decision Trees
        - Ensemble approach: Stacking of classification models
        - May include uplift modeling to predict incremental impact
        
        Key Features:
        - HCP engagement history (past responses to campaigns)
        - Channel preferences (email vs call vs event)
        - Prescribing patterns and brand loyalty
        - HCP demographics (age, years in practice, practice size)
        - Competitive messaging exposure
        - Timing and frequency of outreach
        - Message content and creative elements
        
        Success Metrics:
        - AUC-ROC (Area Under ROC Curve) - higher is better (0.5-1.0)
        - Precision and Recall for response prediction
        - F1 Score for balanced performance
        - Lift and Gain charts for targeting efficiency
        - Uplift metrics for incremental impact
        
        Business Impact:
        - Optimize campaign targeting (who to contact)
        - Personalize message content (what to say)
        - Select best channel (how to reach them)
        - Improve response rates and reduce waste
        - Increase prescription lift from marketing activities
        """,
        "metadata": {
            "typical_algorithms": ["Logistic Regression", "SVM", "Neural Network", "Decision Trees"],
            "ensemble_types": ["stacking", "voting"],
            "key_metrics": ["AUC_ROC", "Precision", "Recall", "F1", "Uplift"],
            "prediction_target": "response_probability",
            "typical_features": ["engagement_history", "channel_preference", "prescribing_patterns", "demographics"]
        },
        "keywords": ["HCP response", "engagement prediction", "campaign optimization", "personalization", "uplift modeling"]
    },
    
    {
        "doc_id": "UC003",
        "category": "use_case",
        "title": "Feature Importance & Driver Analysis",
        "content": """
        Feature Importance Analysis identifies which variables (features) have the 
        greatest impact on model predictions and business outcomes.
        
        Business Context:
        - Answers "What drives HCP prescribing behavior?"
        - Guides marketing strategy by identifying controllable levers
        - Helps prioritize data collection and feature engineering
        - Validates business hypotheses about key drivers
        - Essential for model interpretability and stakeholder trust
        
        Typical Models Used:
        - Tree-based models (Random Forest, XGBoost, LightGBM) with built-in importance
        - SHAP (SHapley Additive exPlanations) for model-agnostic interpretation
        - Permutation importance for unbiased assessment
        - Ensemble models reveal feature interactions and synergies
        
        Importance Types:
        - Gain: Average improvement in objective function when feature is used
        - Split: Number of times feature appears in tree splits
        - SHAP: Contribution of each feature to individual predictions
        - Permutation: Performance drop when feature is shuffled
        - Weight: Feature coefficients (linear models)
        
        Key Insights:
        - Ranking of features by importance score
        - Feature interactions (how features work together)
        - Marginal effects (impact of feature value changes)
        - Non-linear relationships and thresholds
        - Stability of importance across model versions
        
        Business Impact:
        - Focus marketing efforts on high-impact drivers
        - Identify uncontrollable factors (e.g., HCP specialty) vs controllable (e.g., call frequency)
        - Optimize resource allocation across marketing tactics
        - Validate or challenge business assumptions
        - Provide transparency for model-driven decisions
        """,
        "metadata": {
            "typical_algorithms": ["Random Forest", "XGBoost", "LightGBM"],
            "importance_types": ["gain", "split", "shap", "permutation", "weight"],
            "ensemble_advantage": "Reveals feature interactions and synergies",
            "typical_outputs": ["feature_rankings", "interaction_effects", "marginal_plots"]
        },
        "keywords": ["feature importance", "driver analysis", "SHAP", "model interpretation", "business drivers"]
    },
    
    {
        "doc_id": "UC004",
        "category": "use_case",
        "title": "Model Drift Detection",
        "content": """
        Model Drift Detection monitors changes in model performance, data distributions, 
        and prediction patterns over time to identify when models need retraining.
        
        Business Context:
        - HCP behavior evolves (new treatments, guidelines, market dynamics)
        - Data distributions shift (population changes, seasonality)
        - Model performance degrades without monitoring
        - Early detection prevents poor business decisions
        - Critical for production ML systems
        
        Types of Drift:
        1. Concept Drift: Relationship between features and target changes
           - Example: HCP response to messaging changes due to new competitor entry
        
        2. Data Drift: Input feature distributions change
           - Example: Average HCP patient volume decreases
        
        3. Performance Drift: Model accuracy degrades over time
           - Example: RMSE increases from 50 to 75
        
        4. Prediction Drift: Output distributions shift
           - Example: Model predicts fewer high-value HCPs than before
        
        Detection Methods:
        - Statistical tests (KS test, Chi-square, PSI)
        - Performance metric tracking over time
        - Comparing current vs baseline execution metrics
        - Ensemble model consensus analysis
        
        Key Metrics:
        - Drift Score: Magnitude of drift (typically 0-1 scale)
        - Threshold: When to trigger retraining (e.g., 0.10 = 10% change)
        - Affected Features: Which variables are shifting
        - Performance Impact: Metric degradation amount
        
        Business Impact:
        - Prevent acting on stale predictions
        - Trigger timely model retraining
        - Maintain forecast accuracy for planning
        - Detect market shifts early
        - Ensure regulatory compliance (model validation)
        """,
        "metadata": {
            "drift_types": ["concept_drift", "data_drift", "performance_drift", "prediction_drift"],
            "detection_methods": ["statistical_tests", "metric_tracking", "baseline_comparison"],
            "typical_threshold": 0.10,
            "typical_metrics": ["drift_score", "KS_statistic", "PSI", "performance_delta"]
        },
        "keywords": ["model drift", "performance monitoring", "concept drift", "data drift", "retraining triggers"]
    },
    
    {
        "doc_id": "UC005",
        "category": "use_case",
        "title": "Next Best Action & Messaging Optimization",
        "content": """
        Next Best Action (NBA) models recommend the optimal marketing action for each 
        HCP to maximize prescription lift and marketing efficiency.
        
        Business Context:
        - Field force has limited time and resources
        - Different HCPs respond to different messages and channels
        - Goal: Maximize incremental prescriptions per marketing dollar
        - Combines response prediction + uplift modeling + business rules
        
        Decision Framework:
        - Who: Which HCPs to target (prioritization)
        - What: Which message content and creative (personalization)
        - How: Which channel (call, email, event, sample)
        - When: Optimal timing and frequency
        - Why: Expected incremental lift and ROI
        
        Typical Models Used:
        - Uplift models: Predict treatment effect (treated vs control)
        - Response models: Predict engagement likelihood
        - Ensemble combining uplift + response + propensity models
        - Reinforcement learning for sequential decisions
        
        Key Concepts:
        - Uplift Score: Incremental lift from marketing action
        - Control Prediction: Predicted behavior without intervention
        - Treatment Prediction: Predicted behavior with intervention
        - Uplift = Treatment - Control
        - ROI: (Incremental Revenue - Cost) / Cost
        
        Action Prioritization:
        - High uplift HCPs: Will respond positively to messaging
        - Low/negative uplift HCPs: "Sleeping dogs" - don't contact
        - Already convinced HCPs: Will prescribe regardless
        - Lost causes: Won't prescribe no matter what
        
        Business Impact:
        - Increase marketing effectiveness by 20-40%
        - Reduce wasted spend on unresponsive HCPs
        - Personalize customer experience
        - Optimize field force call plans
        - Improve sample allocation efficiency
        """,
        "metadata": {
            "typical_algorithms": ["Uplift Models", "Causal Forests", "Meta-learners"],
            "ensemble_types": ["uplift_ensemble", "meta_learning"],
            "key_metrics": ["uplift_score", "AUUC", "Qini_coefficient", "predicted_ROI"],
            "decisions": ["target_selection", "message_content", "channel", "timing"],
            "typical_features": ["prescribing_potential", "engagement_history", "competitive_pressure", "channel_preference"]
        },
        "keywords": ["next best action", "uplift modeling", "messaging optimization", "personalization", "incremental lift"]
    },
    
    # ========================================================================
    # CATEGORY: ENSEMBLE METHODS
    # ========================================================================
    
    {
        "doc_id": "ENS001",
        "category": "ensemble_method",
        "title": "Ensemble Learning Fundamentals",
        "content": """
        Ensemble learning combines multiple models (base learners) to create a more 
        powerful predictive model than any single model alone.
        
        Core Principle: "Wisdom of Crowds"
        - Different models make different errors
        - Averaging reduces variance and improves stability
        - Combining complementary strengths
        
        When Ensembles Outperform Base Models:
        1. Base models are diverse (different algorithms or training data)
        2. Base models are reasonably accurate (better than random)
        3. Errors are uncorrelated across base models
        4. Problem is complex with non-linear relationships
        5. High-stakes decisions requiring robustness
        
        Why Ensembles May Underperform:
        1. Base models are too similar (no diversity)
        2. One base model is much stronger than others
        3. Overfitting: Ensemble learns noise in training data
        4. Insufficient training data
        5. Poor meta-learner design (stacking)
        6. Excessive complexity without benefit
        
        Pharma Analytics Applications:
        - NRx forecasting: Combine XGBoost + Random Forest + LightGBM
        - HCP response: Stack Logistic Regression + Neural Net + SVM
        - Feature importance: Use ensemble to identify stable drivers
        - Drift detection: Monitor ensemble consensus changes
        
        Key Considerations:
        - Interpretability vs Performance tradeoff
        - Computational cost (training and inference time)
        - Model maintenance complexity
        - Diminishing returns beyond 3-5 base models
        """,
        "metadata": {
            "key_principles": ["diversity", "wisdom_of_crowds", "error_reduction"],
            "success_factors": ["model_diversity", "uncorrelated_errors", "reasonable_accuracy"],
            "failure_modes": ["overfitting", "lack_of_diversity", "poor_meta_learner"],
            "typical_improvements": "5-15% metric improvement over best base model"
        },
        "keywords": ["ensemble learning", "model combination", "wisdom of crowds", "diversity", "variance reduction"]
    },
    
    {
        "doc_id": "ENS002",
        "category": "ensemble_method",
        "title": "Boosting Ensembles",
        "content": """
        Boosting builds an ensemble sequentially, where each new model focuses on 
        correcting the errors of previous models.
        
        How Boosting Works:
        1. Train first model on full dataset
        2. Identify observations with large errors
        3. Give higher weight to misclassified/mispredicted instances
        4. Train next model on reweighted data
        5. Repeat for N iterations
        6. Combine all models with weighted voting
        
        Popular Algorithms:
        - XGBoost: Extreme Gradient Boosting (regularized, parallel)
        - LightGBM: Light Gradient Boosting Machine (fast, efficient)
        - CatBoost: Category Boosting (handles categorical variables well)
        - AdaBoost: Adaptive Boosting (original algorithm)
        
        Strengths:
        - Often achieves best performance on structured data
        - Handles non-linear relationships well
        - Built-in feature importance
        - Less prone to overfitting than single deep trees
        - Excellent for NRx forecasting and response prediction
        
        Weaknesses:
        - Can overfit with too many iterations
        - Sensitive to noisy data and outliers
        - Sequential training (slower than bagging)
        - Less interpretable than single models
        - Requires careful hyperparameter tuning
        
        Pharma Use Cases:
        - NRx forecasting: XGBoost captures seasonal patterns + HCP trends
        - HCP segmentation: LightGBM efficiently handles large HCP databases
        - Territory allocation: Gradient boosting for complex geographic factors
        
        Key Hyperparameters:
        - n_estimators: Number of boosting rounds (50-500)
        - learning_rate: Step size shrinkage (0.01-0.3)
        - max_depth: Tree depth (3-10)
        - min_child_weight: Minimum samples per leaf
        - subsample: Fraction of training data per round
        """,
        "metadata": {
            "learning_approach": "sequential",
            "algorithms": ["XGBoost", "LightGBM", "CatBoost", "AdaBoost"],
            "best_for": ["structured_data", "non_linear_relationships", "tabular_data"],
            "typical_hyperparameters": ["n_estimators", "learning_rate", "max_depth"],
            "risk": "overfitting with too many rounds"
        },
        "keywords": ["boosting", "XGBoost", "LightGBM", "gradient boosting", "sequential learning", "error correction"]
    },
    
    {
        "doc_id": "ENS003",
        "category": "ensemble_method",
        "title": "Bagging & Random Forest",
        "content": """
        Bagging (Bootstrap Aggregating) creates multiple models by training on different 
        random samples of the training data, then averages their predictions.
        
        How Bagging Works:
        1. Create N bootstrap samples (random sampling with replacement)
        2. Train separate model on each sample
        3. For regression: Average all predictions
        4. For classification: Majority vote
        
        Random Forest Enhancement:
        - Each tree uses random subset of features at each split
        - Increases diversity among trees
        - Reduces correlation between base learners
        - Typically 100-500 trees
        
        Strengths:
        - Reduces variance and prevents overfitting
        - Parallel training (faster than boosting)
        - Robust to noisy data and outliers
        - No hyperparameter tuning needed for decent performance
        - Built-in feature importance
        - Handles missing values well
        
        Weaknesses:
        - Can be biased toward majority class (classification)
        - Less effective on linear relationships
        - Larger model size (memory footprint)
        - May not improve much over single tree if data is limited
        
        Pharma Use Cases:
        - NRx forecasting: Robust predictions with confidence intervals
        - HCP clustering: Identify similar prescriber groups
        - Feature importance: Stable rankings across bootstrap samples
        - Outlier detection: Isolate unusual HCP behaviors
        
        Key Hyperparameters:
        - n_estimators: Number of trees (100-500)
        - max_features: Features per split ('sqrt', 'log2', or fraction)
        - max_depth: Maximum tree depth (None or 10-30)
        - min_samples_split: Minimum samples to split node (2-10)
        - bootstrap: Whether to use bootstrap sampling (True)
        
        Comparison to Boosting:
        - Bagging: Parallel, stable, less prone to overfit
        - Boosting: Sequential, higher accuracy, more overfitting risk
        - Random Forest is often first choice for pharma analytics
        """,
        "metadata": {
            "learning_approach": "parallel",
            "algorithms": ["Random Forest", "Extra Trees"],
            "best_for": ["variance_reduction", "robust_predictions", "parallel_training"],
            "typical_hyperparameters": ["n_estimators", "max_features", "max_depth"],
            "advantage": "less overfitting than boosting"
        },
        "keywords": ["bagging", "random forest", "bootstrap aggregating", "variance reduction", "parallel training"]
    },
    
    {
        "doc_id": "ENS004",
        "category": "ensemble_method",
        "title": "Stacking & Meta-Learning",
        "content": """
        Stacking (Stacked Generalization) trains a meta-model to optimally combine 
        predictions from multiple diverse base models.
        
        How Stacking Works:
        1. Train diverse base models (e.g., RF, XGBoost, LR, Neural Net)
        2. Generate predictions from each base model
        3. Use base predictions as features for meta-learner
        4. Train meta-learner to combine base predictions optimally
        5. Final prediction comes from meta-learner
        
        Architecture:
        - Level 0 (Base Layer): 3-5 diverse algorithms
        - Level 1 (Meta Layer): Simple model (LR, Ridge, Elastic Net)
        - Can have multiple levels (deep stacking)
        
        Base Model Selection:
        - Choose diverse algorithms (tree-based + linear + neural)
        - Avoid highly correlated models
        - Each base model should capture different patterns
        - Examples: {Random Forest, XGBoost, Logistic Regression, SVM}
        
        Meta-Learner Selection:
        - Simple models often work best (Linear Regression, Logistic Regression)
        - Regularized models prevent overfitting (Ridge, Lasso, Elastic Net)
        - Neural networks for complex non-linear combinations
        - Should be different from base models
        
        Strengths:
        - Often achieves best performance by leveraging complementary models
        - Learns optimal weighting of base models
        - Can discover non-linear combinations
        - Flexible: Works with any base model types
        - Excellent for high-stakes pharma forecasting
        
        Weaknesses:
        - More complex to implement and maintain
        - Risk of overfitting if not done carefully
        - Requires careful cross-validation
        - Higher computational cost
        - Less interpretable than single models
        
        Best Practices:
        - Use out-of-fold predictions for meta-learner training
        - Apply cross-validation to prevent overfitting
        - Keep meta-learner simple (regularized linear model)
        - Monitor for diminishing returns (>5 base models rarely helps)
        - Track individual base model performance
        
        Pharma Use Cases:
        - NRx forecasting: Combine boosting (trends) + RF (stability) + LR (interpretability)
        - HCP response: Stack multiple classifiers for maximum AUC
        - Territory allocation: Meta-learner weighs regional vs national patterns
        
        Why Stacking May Outperform:
        - Base models capture different aspects of HCP behavior
        - Meta-learner learns when to trust each base model
        - Reduces both bias and variance
        - More robust to unusual market conditions
        """,
        "metadata": {
            "learning_approach": "meta_learning",
            "typical_base_models": ["Random Forest", "XGBoost", "Logistic Regression", "Neural Network"],
            "typical_meta_models": ["Linear Regression", "Ridge", "Elastic Net"],
            "best_for": ["maximum_performance", "leveraging_diversity", "high_stakes_decisions"],
            "risk": "overfitting if not validated properly",
            "recommended_base_count": "3-5 models"
        },
        "keywords": ["stacking", "meta-learning", "model combination", "heterogeneous ensemble", "optimal weighting"]
    },
    
    {
        "doc_id": "ENS005",
        "category": "ensemble_method",
        "title": "Blending vs Stacking",
        "content": """
        Blending is a simpler alternative to stacking that uses a holdout validation 
        set instead of cross-validation for meta-learner training.
        
        Blending Process:
        1. Split data: Train (50%), Validation (25%), Test (25%)
        2. Train base models on Train set
        3. Generate predictions on Validation set
        4. Train meta-learner on Validation predictions
        5. Evaluate on Test set
        
        Stacking Process:
        1. Use full training data
        2. Generate out-of-fold predictions via k-fold CV
        3. Train meta-learner on out-of-fold predictions
        4. Evaluate on separate test set
        
        Comparison:
        
        Blending Advantages:
        - Simpler implementation
        - Faster (no cross-validation)
        - Less prone to overfitting
        - Easier to debug and understand
        - Good for quick prototypes
        
        Stacking Advantages:
        - Uses more training data
        - More robust cross-validation
        - Better performance (typically 1-3% improvement)
        - Standard practice in competitions
        - Better for limited data scenarios
        
        When to Use Blending:
        - Large datasets (>100K samples)
        - Quick iteration needed
        - Computational constraints
        - Team lacks CV expertise
        - Prototyping phase
        
        When to Use Stacking:
        - Limited training data (<50K samples)
        - Maximum performance required
        - Production deployment
        - Time for proper validation
        - Competitive benchmarking
        
        Pharma Recommendation:
        - Start with blending for initial model development
        - Move to stacking for production deployment
        - Use stacking for critical forecasts (launch products)
        - Use blending for routine monthly predictions
        """,
        "metadata": {
            "blending_split": "50% train, 25% validation, 25% test",
            "stacking_approach": "k-fold cross-validation",
            "blending_advantages": ["simpler", "faster", "less_overfitting"],
            "stacking_advantages": ["more_data", "better_performance", "more_robust"],
            "performance_delta": "1-3% in favor of stacking"
        },
        "keywords": ["blending", "stacking comparison", "meta-learning", "holdout validation", "cross-validation"]
    },
    
    # ========================================================================
    # CATEGORY: PERFORMANCE METRICS
    # ========================================================================
    
    {
        "doc_id": "MET001",
        "category": "metric",
        "title": "Regression Metrics for NRx Forecasting",
        "content": """
        Regression metrics evaluate how well models predict continuous values like 
        prescription volumes.
        
        RMSE (Root Mean Squared Error):
        - Formula: sqrt(mean((actual - predicted)²))
        - Units: Same as target variable (e.g., prescriptions)
        - Interpretation: Average prediction error magnitude
        - Penalizes large errors more than small errors
        - Range: 0 to infinity (lower is better)
        - Typical good value: RMSE < 15% of mean target value
        
        Example: If average NRx = 100/month, RMSE = 15 means typical error is 15 prescriptions
        - RMSE = 10: Excellent (10% error)
        - RMSE = 20: Acceptable (20% error)
        - RMSE = 40: Poor (40% error)
        
        MAE (Mean Absolute Error):
        - Formula: mean(|actual - predicted|)
        - Units: Same as target variable
        - Interpretation: Average absolute error
        - Less sensitive to outliers than RMSE
        - Range: 0 to infinity (lower is better)
        - MAE is always ≤ RMSE
        
        Comparison:
        - If RMSE >> MAE: Large outlier errors exist
        - If RMSE ≈ MAE: Errors are consistent (no large outliers)
        
        R² Score (R-Squared / Coefficient of Determination):
        - Formula: 1 - (sum((actual - predicted)²) / sum((actual - mean)²))
        - Units: Dimensionless
        - Interpretation: Proportion of variance explained
        - Range: -infinity to 1 (higher is better)
        - R² = 1.0: Perfect predictions
        - R² = 0.0: Model no better than predicting mean
        - R² < 0.0: Model worse than predicting mean
        
        Typical benchmarks:
        - R² > 0.7: Good model for pharma forecasting
        - R² = 0.5-0.7: Acceptable, room for improvement
        - R² < 0.5: Poor, need better features or model
        
        MAPE (Mean Absolute Percentage Error):
        - Formula: mean(|actual - predicted| / actual) × 100
        - Units: Percentage
        - Interpretation: Average percentage error
        - Range: 0% to infinity (lower is better)
        - Easy to communicate to business stakeholders
        
        Issues with MAPE:
        - Undefined when actual = 0
        - Asymmetric: Penalizes over-predictions more than under-predictions
        - Not suitable when target has zeros or near-zeros
        
        Metric Selection for Pharma:
        - Primary: RMSE (captures large errors, common standard)
        - Secondary: R² (explains variance, easy to interpret)
        - Communication: MAPE (business stakeholders understand %)
        - Robustness check: MAE (verify no extreme outliers)
        
        Ensemble Advantage:
        - Ensembles typically reduce RMSE by 5-15% vs best base model
        - R² improvements of 0.03-0.10 are meaningful
        - More stable predictions (lower variance in RMSE across CV folds)
        """,
        "metadata": {
            "metric_types": ["RMSE", "MAE", "R2", "MAPE", "MSE"],
            "lower_is_better": ["RMSE", "MAE", "MAPE", "MSE"],
            "higher_is_better": ["R2"],
            "typical_primary_metric": "RMSE",
            "business_metric": "MAPE",
            "typical_r2_threshold": 0.7,
            "ensemble_improvement_range": "5-15% RMSE reduction"
        },
        "keywords": ["RMSE", "MAE", "R2", "regression metrics", "forecasting accuracy", "error measurement"]
    },
    
    {
        "doc_id": "MET002",
        "category": "metric",
        "title": "Classification Metrics for HCP Response",
        "content": """
        Classification metrics evaluate how well models predict categorical outcomes 
        like HCP response to marketing campaigns.
        
        AUC-ROC (Area Under Receiver Operating Characteristic Curve):
        - Range: 0.5 to 1.0 (higher is better)
        - Interpretation: Probability model ranks random positive higher than random negative
        - AUC = 0.5: Random guessing
        - AUC = 0.7-0.8: Acceptable discrimination
        - AUC = 0.8-0.9: Good discrimination
        - AUC > 0.9: Excellent (rare in pharma, check for data leakage)
        
        Why AUC-ROC is preferred in pharma:
        - Threshold-independent (evaluates ranking ability)
        - Robust to class imbalance
        - Easy to compare models
        - Industry standard for response modeling
        
        Typical pharma benchmarks:
        - HCP response prediction: AUC = 0.75-0.85
        - NRx/No-NRx classification: AUC = 0.70-0.80
        - Event attendance: AUC = 0.65-0.75
        
        Accuracy:
        - Formula: (TP + TN) / Total
        - Range: 0 to 1 (higher is better)
        - Problem: Misleading with imbalanced classes
        
        Example: If only 5% of HCPs respond to campaigns:
        - Model predicting "no response" for everyone: 95% accuracy
        - But completely useless for business!
        - AUC would correctly show ~0.5 (random)
        
        Precision (Positive Predictive Value):
        - Formula: TP / (TP + FP)
        - Interpretation: Of predicted responders, what % actually respond?
        - Important when cost of false positives is high
        - Pharma use: Minimize wasted marketing spend on predicted responders who don't respond
        
        Recall (Sensitivity, True Positive Rate):
        - Formula: TP / (TP + FN)
        - Interpretation: Of actual responders, what % did we identify?
        - Important when cost of false negatives is high
        - Pharma use: Ensure we don't miss high-potential HCPs
        
        F1 Score:
        - Formula: 2 × (Precision × Recall) / (Precision + Recall)
        - Range: 0 to 1 (higher is better)
        - Harmonic mean of precision and recall
        - Use when you need balance between precision and recall
        
        Precision-Recall Tradeoff:
        - High precision → Few false alarms, may miss opportunities
        - High recall → Capture all opportunities, more false alarms
        - Business decision depends on cost structure
        
        Lift and Gain Charts:
        - Lift: How much better than random targeting
        - Top decile lift: Response rate in top 10% / overall response rate
        - Typical good lift: 3-5x in top decile
        - Used for campaign targeting optimization
        
        Confusion Matrix Elements:
        - True Positive (TP): Correctly predicted responder
        - True Negative (TN): Correctly predicted non-responder
        - False Positive (FP): Predicted responder, actually didn't respond (wasted spend)
        - False Negative (FN): Predicted non-responder, actually would respond (missed opportunity)
        
        Metric Selection for Pharma:
        - Primary: AUC-ROC (standard for response models)
        - Secondary: Precision at top K% (for targeting efficiency)
        - Tertiary: Lift charts (for business communication)
        - Avoid: Raw accuracy (misleading with imbalance)
        
        Ensemble Advantage:
        - Ensembles typically improve AUC by 0.02-0.05
        - Even 0.02 AUC improvement = significant $ in large campaigns
        - More stable predictions across different time periods
        """,
        "metadata": {
            "metric_types": ["AUC_ROC", "Precision", "Recall", "F1", "Accuracy", "Lift"],
            "primary_metric": "AUC_ROC",
            "typical_auc_range": "0.70-0.85",
            "good_lift": "3-5x in top decile",
            "ensemble_improvement": "0.02-0.05 AUC increase",
            "avoid": "accuracy with imbalanced classes"
        },
        "keywords": ["AUC-ROC", "precision", "recall", "F1 score", "classification metrics", "response modeling", "lift"]
    },
    
    {
        "doc_id": "MET003",
        "category": "metric",
        "title": "Uplift Modeling Metrics",
        "content": """
        Uplift modeling metrics measure a model's ability to identify individuals 
        who will respond positively to treatment (marketing intervention).
        
        Key Concept:
        - Goal: Maximize incremental impact, not just response rate
        - Four segments:
          1. Persuadables: Respond only if treated (TARGET THESE)
          2. Sure Things: Respond regardless (don't waste resources)
          3. Lost Causes: Don't respond regardless (avoid)
          4. Sleeping Dogs: Respond only if NOT treated (definitely avoid)
        
        AUUC (Area Under Uplift Curve):
        - Range: -0.5 to 0.5 (higher is better)
        - AUUC = 0: No uplift capability (random)
        - AUUC > 0.05: Good uplift model for pharma
        - AUUC > 0.10: Excellent uplift model
        - Interpretation: Expected incremental response from optimal targeting
        
        Qini Coefficient:
        - Variant of AUUC with different weighting
        - Range: Unbounded (higher is better)
        - More sensitive to gains in high-scoring individuals
        - Preferred in academic literature
        
        Uplift at Top K%:
        - Incremental response rate in top K% vs random
        - Example: Top 20% has 15% uplift, random has 5% uplift
        - Uplift at top 20% = 15% - 5% = 10 percentage points
        - Directly translates to ROI
        
        Incremental Lift:
        - Treatment effect size
        - Formula: (Response rate treated - Response rate control) / Response rate control
        - Example: 10% treated response, 6% control response
        - Incremental lift = (10% - 6%) / 6% = 67% improvement
        
        Practical Pharma Example:
        Without uplift model (random targeting):
        - Contact 10,000 HCPs
        - 500 respond (5% response rate)
        - Cost: $50,000
        - Revenue: $100,000
        - ROI: 100%
        
        With uplift model (target persuadables):
        - Contact 10,000 HCPs (top 20% by uplift score)
        - 800 respond (8% response rate in targeted group)
        - Cost: $50,000
        - Revenue: $160,000
        - ROI: 220%
        - Improvement: 60 percentage points ROI
        
        Challenges:
        - Requires randomized control group data
        - More complex than standard response models
        - Needs larger sample sizes
        - Difficult to validate in production
        
        Model Requirements:
        - Treatment and control group observations
        - Sufficient overlap in covariate distributions
        - Representative randomization
        - Minimum sample size: 10,000+ per group
        
        Ensemble Advantage for Uplift:
        - Combining causal forest + meta-learners reduces bias
        - More robust to treatment effect heterogeneity
        - Better handling of small treatment effects
        - Typical improvement: 0.02-0.04 AUUC
        """,
        "metadata": {
            "metric_types": ["AUUC", "Qini", "uplift_at_top_k", "incremental_lift"],
            "typical_auuc_range": "0.02-0.10",
            "good_auuc": 0.05,
            "requires": "randomized control group",
            "four_segments": ["persuadables", "sure_things", "lost_causes", "sleeping_dogs"],
            "target_segment": "persuadables"
        },
        "keywords": ["uplift modeling", "AUUC", "Qini", "incremental lift", "treatment effect", "causal inference", "persuadables"]
    },
    
    # ========================================================================
    # CATEGORY: BUSINESS CONTEXT
    # ========================================================================
    
    {
        "doc_id": "BIZ001",
        "category": "business_context",
        "title": "Pharmaceutical Commercial Model Lifecycle",
        "content": """
        ML models in pharma commercial analytics go through a structured lifecycle 
        from development to retirement.
        
        1. Model Development (Weeks 1-8):
        - Business problem definition and success metrics
        - Data collection and quality assessment
        - Feature engineering with commercial input
        - Algorithm selection and hyperparameter tuning
        - Cross-validation and performance evaluation
        - Model comparison (base models vs ensembles)
        - Stakeholder review and approval
        
        Key deliverables:
        - Model performance report
        - Feature importance analysis
        - Business case and expected ROI
        - Deployment plan
        
        2. Model Deployment (Weeks 9-12):
        - Integration with data pipelines
        - Prediction generation workflow
        - Business rule integration
        - User interface development
        - Training for commercial teams
        - Production monitoring setup
        
        3. Model Monitoring (Ongoing):
        - Weekly: Prediction volume and distribution checks
        - Monthly: Performance metrics vs baseline
        - Quarterly: Drift detection analysis
        - Annual: Full model revalidation
        
        Monitoring triggers:
        - Performance degradation > 10%
        - Data drift score > threshold
        - Prediction distribution shifts
        - Business outcomes diverge from predictions
        - New competitive entry or market change
        
        4. Model Retraining (As needed):
        - Triggered by monitoring alerts
        - Incorporate recent data (rolling window)
        - Re-evaluate feature importance
        - Compare new version to current version
        - A/B testing before full deployment
        
        Retraining frequency:
        - NRx models: Monthly or quarterly
        - HCP response: After each major campaign
        - Uplift models: Bi-annually
        - Drift detection: Continuous monitoring
        
        5. Model Retirement (When appropriate):
        - New model significantly outperforms (>15% improvement)
        - Business use case changes
        - Regulatory requirements change
        - Data sources deprecated
        - Market dynamics fundamentally shift
        
        Version Control:
        - Semantic versioning (v1.0, v1.1, v2.0)
        - Major version: Algorithm or architecture change
        - Minor version: Feature updates or hyperparameter tuning
        - Patch: Bug fixes or minor adjustments
        - All versions tracked in model registry
        
        Governance:
        - Model risk assessment
        - Documentation requirements
        - Approval workflows
        - Audit trail for predictions
        - Regulatory compliance (where applicable)
        
        Ensemble-Specific Considerations:
        - Track performance of individual base models
        - Monitor meta-learner weights over time
        - More complex deployment infrastructure
        - Longer training times
        - Higher maintenance overhead
        - Document ensemble composition and rationale
        """,
        "metadata": {
            "phases": ["development", "deployment", "monitoring", "retraining", "retirement"],
            "monitoring_frequency": {"weekly": "volume_checks", "monthly": "performance", "quarterly": "drift", "annual": "revalidation"},
            "retraining_triggers": ["performance_degradation", "data_drift", "market_change"],
            "version_types": ["major", "minor", "patch"]
        },
        "keywords": ["model lifecycle", "deployment", "monitoring", "retraining", "governance", "version control"]
    },
    
    {
        "doc_id": "BIZ002",
        "category": "business_context",
        "title": "Pharma Commercial Terminology",
        "content": """
        Key terminology used in pharmaceutical commercial analytics and modeling.
        
        PRESCRIPTION METRICS:
        
        TRx (Total Prescriptions):
        - New prescriptions + Refills
        - Total volume of prescriptions written
        - Lagging indicator (includes ongoing treatment)
        
        NRx (New Prescriptions):
        - Only new prescriptions (first fill)
        - Leading indicator of market adoption
        - More sensitive to marketing efforts
        - Critical for product launches
        
        NBRx (New-to-Brand Prescriptions):
        - New prescriptions from HCPs who haven't prescribed brand before
        - Indicator of market expansion
        
        Refill Rate:
        - Percentage of NRx that become refills
        - Indicator of treatment persistence
        - TRx/NRx ratio
        
        HEALTHCARE PROVIDERS (HCPs):
        
        HCP:
        - Healthcare Provider (doctor, nurse practitioner, physician assistant)
        - Primary target for pharma marketing
        
        Prescriber Types:
        - High prescribers: Top 10-20% by volume
        - Medium prescribers: Middle 30-40%
        - Low prescribers: Bottom 40-50%
        - Non-prescribers: Have not written prescriptions
        
        HCP Specialty:
        - Therapeutic area focus (e.g., cardiology, oncology, primary care)
        - Major segmentation dimension
        - Different specialties have different prescribing patterns
        
        Decile Segmentation:
        - HCPs ranked into 10 equal groups by prescription volume
        - Decile 1: Top 10% of prescribers
        - Decile 10: Bottom 10% of prescribers
        - Marketing often focuses on deciles 1-3
        
        MARKETING ACTIVITIES:
        
        Detailing:
        - Face-to-face sales calls with HCPs
        - Most expensive marketing channel
        - ~15-30 minutes per call
        - Goal: Education and relationship building
        
        Call Plan:
        - Sales rep's schedule of HCP visits
        - Optimized by territory and priority
        - Typical frequency: Monthly or quarterly per HCP
        
        Sample Drops:
        - Free product samples provided to HCPs
        - For patient trials
        - Significant cost, closely tracked
        
        Speaker Programs:
        - HCPs paid to present to peers
        - High engagement, expensive
        - Regulated by compliance
        
        Digital Channels:
        - Email campaigns
        - Banner ads
        - Educational webinars
        - Lower cost, measurable engagement
        
        MARKET METRICS:
        
        Share of Voice (SOV):
        - Brand's marketing presence vs competitors
        - Measured by call frequency, spend, impressions
        
        Market Share:
        - Brand's prescriptions / Total category prescriptions
        - Key success metric
        
        Patient Share:
        - Percentage of patients on brand vs alternatives
        - More stable than script share
        
        TERRITORY MANAGEMENT:
        
        Territory:
        - Geographic sales region assigned to rep
        - Typically 100-300 HCPs per territory
        - Optimized by potential, not just geography
        
        Alignment:
        - Process of dividing market into territories
        - Goal: Balance workload and opportunity
        
        Targeting:
        - Selecting which HCPs to prioritize
        - Based on potential, accessibility, alignment
        
        MODELING TERMINOLOGY:
        
        Lookalike Modeling:
        - Finding HCPs similar to known high responders
        - Clustering and similarity algorithms
        
        Propensity Score:
        - Likelihood of desired behavior (prescribe, respond, attend)
        - Output of response models
        - Used for targeting and ranking
        
        Attribution:
        - Assigning credit for prescriptions to marketing activities
        - Challenge: Multiple touchpoints
        - Multi-touch attribution models
        
        Baseline Forecast:
        - Expected prescriptions without incremental marketing
        - Control group or time-series projection
        - Used to calculate uplift
        
        Incrementality:
        - Prescriptions caused by marketing above baseline
        - Uplift modeling target
        - Key ROI metric
        """,
        "metadata": {
            "prescription_types": ["TRx", "NRx", "NBRx"],
            "hcp_segments": ["high_prescribers", "medium_prescribers", "low_prescribers"],
            "marketing_channels": ["detailing", "samples", "speakers", "digital"],
            "key_metrics": ["market_share", "share_of_voice", "incrementality"]
        },
        "keywords": ["TRx", "NRx", "HCP", "detailing", "market share", "share of voice", "incrementality", "pharma terminology"]
    },
    
    {
        "doc_id": "BIZ003",
        "category": "business_context",
        "title": "Why Ensembles Fail: Common Pitfalls",
        "content": """
        Understanding when and why ensemble models underperform helps avoid 
        costly mistakes in pharma commercial analytics.
        
        PITFALL 1: Lack of Base Model Diversity
        Problem:
        - Using multiple similar algorithms (e.g., 3 versions of XGBoost)
        - All base models make the same errors
        - No benefit from combination
        
        Example:
        - Base models: XGBoost, LightGBM, CatBoost (all gradient boosting)
        - Ensemble RMSE: 45
        - Best base RMSE: 46
        - Improvement: Only 2% (not worth complexity)
        
        Solution:
        - Mix algorithm families: Trees + Linear + Neural
        - Use different feature sets for each base model
        - Vary training data (bagging) or approach (boosting)
        
        PITFALL 2: One Dominant Base Model
        Problem:
        - One base model much stronger than others
        - Weak models add noise, not signal
        - Ensemble performs worse than best single model
        
        Example:
        - XGBoost RMSE: 40 (strong)
        - Linear Regression RMSE: 80 (weak)
        - Random Forest RMSE: 75 (weak)
        - Ensemble RMSE: 42 (worse than XGBoost alone!)
        
        Solution:
        - Only include models within 10-15% of best performer
        - Remove or improve weak base models
        - Consider weighted voting (more weight to strong models)
        - Use stacking to learn optimal weights automatically
        
        PITFALL 3: Overfitting in Meta-Learner
        Problem:
        - Meta-learner too complex (e.g., deep neural network)
        - Learns noise in validation set
        - Poor generalization to new data
        
        Example:
        - Validation R²: 0.85 (great!)
        - Test R²: 0.65 (poor!)
        - Gap indicates overfitting
        
        Solution:
        - Use simple meta-learner (linear regression, ridge)
        - Apply regularization (L1/L2 penalty)
        - Use out-of-fold predictions for training
        - Monitor validation vs test performance gap
        
        PITFALL 4: Data Leakage
        Problem:
        - Future information leaked into model training
        - Unrealistic performance in development
        - Catastrophic failure in production
        
        Common sources in pharma:
        - Using future prescription data to predict past
        - Including target-derived features
        - Not respecting temporal ordering
        - Using post-campaign data to predict campaign response
        
        Example:
        - Model achieves R² = 0.95 (suspiciously high!)
        - Investigation reveals "total_annual_prescriptions" feature
        - This includes future data not available at prediction time
        - Production performance: R² = 0.60
        
        Solution:
        - Strict temporal validation splits
        - Feature engineering audit
        - Production simulation during development
        - Monitoring for performance cliff in production
        
        PITFALL 5: Insufficient Training Data
        Problem:
        - Not enough data to train multiple models + meta-learner
        - Ensembles need more data than single models
        - High variance in performance estimates
        
        Rule of thumb:
        - Minimum: 10,000 observations for simple ensemble
        - Recommended: 50,000+ for complex stacking
        - More data needed for many features or complex meta-learner
        
        Solution:
        - Use simpler ensemble (bagging vs stacking)
        - Reduce number of base models
        - Consider single strong model instead
        - Collect more data before deploying ensemble
        
        PITFALL 6: Ignoring Business Context
        Problem:
        - Optimizing pure statistical metrics
        - Ignoring business costs and constraints
        - Model not aligned with business goals
        
        Example:
        - Ensemble AUC: 0.82 vs Best Base AUC: 0.80
        - Ensemble training time: 10 hours vs Base: 1 hour
        - Ensemble complexity: 5 models vs Base: 1 model
        - Business impact: Minimal (predictions used monthly, not real-time)
        - Decision: Not worth ensemble complexity for 0.02 AUC gain
        
        Solution:
        - Define business-relevant metrics
        - Consider maintenance costs
        - Evaluate complexity vs benefit tradeoff
        - Prototype and measure ROI before full deployment
        
        PITFALL 7: Poor Cross-Validation Strategy
        Problem:
        - Random CV on time-series data
        - Data leakage across folds
        - Overly optimistic performance estimates
        
        Solution for pharma:
        - Use time-series CV (train on past, validate on future)
        - Block/Group CV for hierarchical data (HCPs within territories)
        - Stratified CV to maintain class balance
        - Leave-one-group-out for generalization testing
        
        PITFALL 8: Not Monitoring Individual Base Models
        Problem:
        - Only tracking ensemble performance
        - Don't know which base models are drifting
        - Can't diagnose issues
        
        Solution:
        - Track all base model metrics separately
        - Monitor meta-learner weights over time
        - Alert when base model performance diverges significantly
        - Regular model health checks
        
        Warning Signs Your Ensemble Is Failing:
        - Minimal improvement over best base model (<3%)
        - High variance in cross-validation scores
        - Large gap between validation and test performance
        - Base models highly correlated (>0.95)
        - Meta-learner weights very imbalanced (e.g., 0.95 on one model)
        - Predictions don't make business sense
        - Maintenance overhead exceeds benefit
        """,
        "metadata": {
            "common_pitfalls": [
                "lack_of_diversity",
                "one_dominant_model",
                "meta_learner_overfitting",
                "data_leakage",
                "insufficient_data",
                "ignoring_business_context",
                "poor_cv_strategy"
            ],
            "minimum_data": 10000,
            "recommended_data": 50000,
            "minimum_improvement_threshold": "3-5%",
            "warning_signs": ["low_improvement", "high_variance", "val_test_gap"]
        },
        "keywords": ["ensemble failures", "pitfalls", "overfitting", "data leakage", "model diversity", "common mistakes"]
    },
    
    # ========================================================================
    # CATEGORY: FEATURES & DATA
    # ========================================================================
    
    {
        "doc_id": "FEAT001",
        "category": "features",
        "title": "HCP Features for Prescribing Models",
        "content": """
        Key features used to predict HCP prescribing behavior in pharma analytics.
        
        DEMOGRAPHIC FEATURES:
        
        Specialty:
        - Primary therapeutic focus (cardiology, oncology, primary care, etc.)
        - Most important feature (typically top 3)
        - High predictive power for product-specific models
        - Categorical encoding: One-hot or target encoding
        
        Years in Practice:
        - Time since medical school graduation
        - Proxy for experience and prescribing patterns
        - Younger HCPs may be more open to new treatments
        - Older HCPs may have established prescribing habits
        
        Practice Setting:
        - Hospital, clinic, private practice, academic medical center
        - Influences access and prescribing autonomy
        - Hospital settings may have formulary restrictions
        
        Practice Size:
        - Number of providers in group
        - Solo practitioners vs large groups
        - Impacts decision-making process
        
        Patient Volume:
        - Patients seen per month/year
        - Direct relationship with prescription potential
        - Often top 5 most important feature
        
        Geography:
        - ZIP code, city, state, region
        - Captures local market dynamics
        - Regional prescribing variations
        
        PRESCRIBING HISTORY FEATURES:
        
        Historical TRx/NRx:
        - Past prescription volumes (6-12 month lookback)
        - Strongest predictor (often #1 feature)
        - Trend: Increasing, stable, or declining
        
        Brand Loyalty:
        - Percentage of prescriptions for specific brand
        - Indicates switching likelihood
        - Calculated per therapeutic class
        
        Competitive Prescribing:
        - Prescriptions for competitor products
        - Market share within HCP practice
        - Switching potential indicator
        
        Therapeutic Class Experience:
        - Total prescriptions in relevant therapeutic area
        - Indicates treatment familiarity
        - Proxy for patient population
        
        Prescribing Breadth:
        - Number of different products prescribed
        - Early adopter indicator
        - Risk tolerance proxy
        
        Seasonality Patterns:
        - Month-over-month variation
        - Captures cyclical trends
        - Important for accurate forecasting
        
        MARKETING TOUCHPOINT FEATURES:
        
        Call Frequency:
        - Number of sales rep visits (past 3-6 months)
        - Lag effect: Impact may be delayed 1-2 months
        - Diminishing returns after 3-4 calls per quarter
        
        Sample Volume:
        - Number of samples received
        - Strong predictor for trial initiation
        - Regulatory limits vary by product
        
        Digital Engagement:
        - Email opens, clicks, website visits
        - Lower cost touchpoint
        - Engagement score composite
        
        Event Attendance:
        - Speaker programs, conferences, dinners
        - High engagement signal
        - Often combined with peer influence
        
        Last Touchpoint Recency:
        - Days since last interaction
        - Decay function for impact
        - Typical half-life: 30-60 days
        
        Touchpoint Sequencing:
        - Order and combination of channels
        - Multi-touch attribution
        - Synergy between channels
        
        Message Exposure:
        - Specific campaigns or messages seen
        - A/B testing of creative
        - Resonance varies by HCP segment
        
        MARKET & COMPETITIVE FEATURES:
        
        Share of Voice:
        - Brand marketing presence vs competitors
        - Territory-level or HCP-level
        - Includes all channels
        
        Competitive Activity:
        - Competitor sales rep visits
        - Competitive sample drops
        - Market entry/exit events
        
        Local Market Share:
        - Brand share in HCP's geography
        - Peer prescribing influence
        - Network effects
        
        Formulary Status:
        - Coverage tier and restrictions
        - Prior authorization requirements
        - Patient access barriers
        
        PATIENT POPULATION FEATURES:
        
        Patient Demographics:
        - Age distribution of patients
        - Insurance mix (commercial, Medicare, Medicaid)
        - Socioeconomic indicators
        
        Comorbidity Profile:
        - Prevalence of relevant conditions
        - Treatment complexity
        - Polypharmacy considerations
        
        Patient Turnover:
        - New patient rate
        - Patient retention
        - Practice growth indicator
        
        FEATURE ENGINEERING TIPS:
        
        Lag Features:
        - Use 1, 3, 6, 12 month lags of prescription volumes
        - Capture momentum and trends
        - Example: NRx_lag_1, NRx_lag_3
        
        Rolling Statistics:
        - Moving averages (3, 6, 12 months)
        - Standard deviations (volatility)
        - Min/max ranges
        
        Ratios and Interactions:
        - NRx/Patient_volume (penetration rate)
        - Call_frequency * Sample_volume (intensity)
        - Brand_share / Competitor_share (relative position)
        
        Time-based Features:
        - Month, quarter (seasonality)
        - Days since launch
        - Time since last prescription
        
        Segment-based Features:
        - HCP tier (high/medium/low potential)
        - Behavioral segment (early adopter, conservative, etc.)
        - Risk group
        
        FEATURE IMPORTANCE IN PHARMA ENSEMBLES:
        
        Typical Top 10 Features (NRx Forecasting):
        1. Historical NRx (lag 1-3 months)
        2. Patient volume
        3. HCP specialty
        4. Call frequency (lag 1 month)
        5. Historical TRx trends
        6. Geographic region
        7. Sample volume
        8. Competitive prescribing
        9. Practice setting
        10. Years in practice
        
        Ensemble-Specific Insights:
        - Boosting models favor historical prescribing (70% importance)
        - Linear models favor demographic and marketing features
        - Neural networks find complex interactions
        - Ensemble combines all perspectives
        - Meta-learner learns when to trust each model
        """,
        "metadata": {
            "feature_categories": [
                "demographic",
                "prescribing_history",
                "marketing_touchpoints",
                "market_competitive",
                "patient_population"
            ],
            "top_features": [
                "historical_nrx",
                "patient_volume",
                "specialty",
                "call_frequency",
                "trx_trends"
            ],
            "typical_feature_count": "50-200 features",
            "lag_periods": [1, 3, 6, 12]
        },
        "keywords": ["features", "HCP attributes", "prescribing behavior", "marketing touchpoints", "feature engineering", "predictive variables"]
    },
    
    {
        "doc_id": "FEAT002",
        "category": "features",
        "title": "Feature Interactions in Ensemble Models",
        "content": """
        Feature interactions occur when the combined effect of two features differs 
        from their individual effects. Ensemble models excel at capturing these.
        
        WHY INTERACTIONS MATTER:
        
        Real-world behavior is non-linear:
        - Marketing impact depends on HCP characteristics
        - Specialty determines which features are important
        - Historical behavior moderates marketing effectiveness
        
        Example Interaction:
        - Call frequency alone: Low importance
        - HCP engagement score alone: Medium importance
        - Call frequency × Engagement: High importance
        - Interpretation: Calls only work on engaged HCPs
        
        COMMON PHARMA INTERACTIONS:
        
        1. Specialty × Marketing Channel
        - Primary care HCPs respond to different channels than specialists
        - Cardiologists prefer conferences, PCPs prefer samples
        - Model learns specialty-specific marketing mix
        
        2. Historical Prescribing × Call Frequency
        - High historical prescribers: Minimal call impact (already prescribing)
        - Medium prescribers: Strong call impact (growth potential)
        - Low prescribers: Moderate call impact (awareness building)
        - Interaction reveals targeting sweet spot
        
        3. Patient Volume × Sample Drops
        - High patient volume + High samples = Strong effect
        - High patient volume + Low samples = Missed opportunity
        - Low patient volume + High samples = Wasted resources
        - Optimal sample allocation per HCP volume
        
        4. Competitive Pressure × Brand Loyalty
        - High competition + Low loyalty = High switching risk
        - High competition + High loyalty = Defensive success
        - Low competition + Low loyalty = Growth opportunity
        - Guides competitive strategy
        
        5. Geography × Seasonality
        - Northern states: Flu season prescribing peaks
        - Southern states: Different seasonal patterns
        - Regional campaigns timed to local patterns
        
        6. Years in Practice × Early Adopter Behavior
        - Young HCPs + Early adopter = Trial-prone (high potential)
        - Young HCPs + Conservative = Need education
        - Experienced HCPs + Early adopter = Opinion leaders
        - Experienced HCPs + Conservative = Difficult to change
        
        HOW ENSEMBLES CAPTURE INTERACTIONS:
        
        Tree-Based Models:
        - Natural interaction detection through splits
        - Example: Split on specialty, then split on call frequency
        - Automatic feature interaction up to tree depth
        
        Linear Models:
        - Require explicit interaction terms
        - Feature_A * Feature_B
        - Combinatorial explosion with many features
        
        Neural Networks:
        - Hidden layers learn interactions
        - Black box nature makes interpretation difficult
        - Can capture very complex interactions
        
        Ensemble Advantage:
        - Trees capture local interactions
        - Linear models capture global trends
        - Neural nets capture complex non-linear patterns
        - Meta-learner combines complementary interaction patterns
        - Result: More robust and complete interaction coverage
        
        DETECTING IMPORTANT INTERACTIONS:
        
        SHAP Interaction Values:
        - Quantifies pairwise feature interactions
        - Shows how features work together
        - Visualize with interaction plots
        
        Partial Dependence Plots:
        - Show effect of feature combinations
        - 2D plots reveal interaction surfaces
        - Example: Call frequency on X, Historical NRx on Y, Color = Prediction
        
        H-Statistic:
        - Measures strength of pairwise interactions
        - 0 = No interaction, 1 = Pure interaction
        - Identifies which interactions to investigate
        
        Business Rules from Interactions:
        - "Increase call frequency for medium historical prescribers in cardiology"
        - "Allocate more samples to high patient volume PCPs"
        - "Focus digital channels on younger, tech-savvy HCPs"
        
        INTERACTION EXAMPLES WITH NUMBERS:
        
        Example 1: Specialty × Marketing
        Primary Care:
        - Base response rate: 5%
        - With samples: +8% (13% total)
        - With calls: +3% (8% total)
        - Sample interaction effect: Strong
        
        Cardiology:
        - Base response rate: 8%
        - With samples: +2% (10% total)
        - With calls: +7% (15% total)
        - Call interaction effect: Strong
        
        Business action: Customize channel mix by specialty
        
        Example 2: Historical NRx × Call Frequency
        High Historical (>50 NRx/month):
        - 0 calls: 55 NRx predicted
        - 4 calls: 57 NRx predicted
        - Marginal gain: 2 NRx (4% lift)
        
        Medium Historical (10-50 NRx/month):
        - 0 calls: 20 NRx predicted
        - 4 calls: 32 NRx predicted
        - Marginal gain: 12 NRx (60% lift)
        
        Low Historical (<10 NRx/month):
        - 0 calls: 3 NRx predicted
        - 4 calls: 5 NRx predicted
        - Marginal gain: 2 NRx (67% lift)
        
        Business action: Prioritize calls to medium prescribers (best absolute gain)
        
        ENSEMBLE VS BASE MODEL INTERACTION HANDLING:
        
        Single Decision Tree:
        - Captures up to depth-level interactions
        - Example: Depth 5 = up to 5-way interactions
        - Limited generalization
        
        Random Forest:
        - Averages many trees
        - Stable interaction detection
        - May miss weak interactions
        
        Gradient Boosting:
        - Sequential error correction
        - Excellent for strong interactions
        - Can overfit to spurious interactions
        
        Ensemble (Stacking):
        - Combines all interaction patterns
        - Random Forest: Robust main effects
        - XGBoost: Strong local interactions
        - Linear model: Global trends
        - Meta-learner: Optimal weighting by context
        - Result: 10-20% better interaction coverage
        """,
        "metadata": {
            "common_interactions": [
                "specialty_marketing_channel",
                "historical_prescribing_calls",
                "patient_volume_samples",
                "competitive_pressure_loyalty",
                "geography_seasonality"
            ],
            "detection_methods": ["SHAP_interaction", "partial_dependence", "h_statistic"],
            "ensemble_advantage": "combines multiple interaction patterns"
        },
        "keywords": ["feature interactions", "non-linear effects", "SHAP", "synergy", "combined effects", "interaction detection"]
    },
    
    # ========================================================================
    # CATEGORY: TROUBLESHOOTING & DEBUGGING
    # ========================================================================
    
    {
        "doc_id": "DEBUG001",
        "category": "troubleshooting",
        "title": "Diagnosing Ensemble Performance Issues",
        "content": """
        Systematic approach to troubleshooting when ensemble models underperform 
        expectations in pharma commercial analytics.
        
        STEP 1: COMPARE TO BASELINE
        
        Questions:
        - What is ensemble performance vs best base model?
        - What is expected improvement range? (typically 3-10%)
        - Is improvement statistically significant?
        - Is improvement consistent across validation folds?
        
        Red flags:
        - Ensemble worse than best base model
        - Ensemble improvement <2%
        - High variance across CV folds
        - Improvement disappears on test set
        
        STEP 2: CHECK BASE MODEL DIVERSITY
        
        Metrics:
        - Prediction correlation between base models
        - Should be <0.9 for good diversity
        - If >0.95, models too similar
        
        Analysis:
        ```
        Check pairwise correlations:
        - Model A vs Model B: 0.97 (too high!)
        - Model A vs Model C: 0.85 (good)
        - Model B vs Model C: 0.92 (borderline)
        
        Action: Remove Model B or replace with different algorithm
        ```
        
        STEP 3: EXAMINE META-LEARNER WEIGHTS (Stacking)
        
        Balanced weights (healthy ensemble):
        - Model A: 0.35
        - Model B: 0.40
        - Model C: 0.25
        - All models contributing
        
        Imbalanced weights (problem):
        - Model A: 0.85
        - Model B: 0.10
        - Model C: 0.05
        - Action: Just use Model A, ensemble adds complexity without benefit
        
        STEP 4: VALIDATE DATA SPLITS
        
        Check for:
        - Time-series leakage (future data in training)
        - Group leakage (HCPs in both train and validation)
        - Target leakage (features derived from target)
        
        Validation strategy for pharma:
        - Time-series CV: Train on months 1-12, validate on 13-15, test on 16-18
        - Block CV: Ensure HCP-level separation
        - Stratified CV: Maintain specialty distribution
        
        STEP 5: ANALYZE ERROR PATTERNS
        
        Where does ensemble fail?
        - Error by HCP specialty
        - Error by prescription volume range
        - Error by geography
        - Error by time period
        
        Example analysis:
        ```
        Specialty Error Analysis:
        - Primary Care: RMSE = 12 (good)
        - Cardiology: RMSE = 18 (acceptable)
        - Oncology: RMSE = 45 (poor!)
        
        Investigation: Oncology data sparse, need separate model
        ```
        
        STEP 6: CHECK FOR OVERFITTING
        
        Signs:
        - Training metrics much better than validation
        - Large gap between CV and test performance
        - Performance degrades in production
        
        Example:
        ```
        Training R²: 0.88
        Validation R²: 0.75
        Test R²: 0.71
        Gap: 0.17 (concerning, indicates overfitting)
        
        Solutions:
        - Reduce meta-learner complexity
        - Add regularization
        - Increase training data
        - Reduce number of base models
        ```
        
        STEP 7: INVESTIGATE FEATURE IMPORTANCE STABILITY
        
        Healthy ensemble:
        - Top 10 features consistent across base models
        - Ranking order may differ, but same features appear
        - Stable across CV folds
        
        Problem ensemble:
        - Different top features in each base model
        - Top features change dramatically across folds
        - Suggests insufficient data or poor features
        
        STEP 8: BUSINESS VALIDATION
        
        Questions:
        - Do predictions make business sense?
        - Are predictions actionable?
        - What do subject matter experts think?
        - Are there obvious errors?
        
        Example red flags:
        - Predicting 1000 NRx for HCP who historically writes 5
        - Negative predictions
        - Predictions outside reasonable ranges
        - Counter-intuitive feature importance
        
        COMMON ROOT CAUSES:
        
        1. Insufficient Training Data
        - Symptoms: High variance, unstable feature importance
        - Minimum: 10,000 samples for simple ensemble
        - Solution: Collect more data or simplify model
        
        2. Poor Feature Engineering
        - Symptoms: Low R² across all models, important features missing
        - Solution: Add domain knowledge, create lag features, interactions
        
        3. Data Quality Issues
        - Symptoms: Outliers, missing values, inconsistent definitions
        - Check: Data profiling, outlier detection, missingness patterns
        - Solution: Data cleaning pipeline, imputation strategy
        
        4. Model Mismatch
        - Symptoms: Linear model performs best for non-linear problem
        - Solution: Use appropriate algorithm family
        
        5. Hyperparameter Tuning
        - Symptoms: Default parameters, no optimization
        - Solution: Grid search or Bayesian optimization
        
        6. Concept Drift
        - Symptoms: Model worked before, now failing
        - Check: Performance over time, distribution shifts
        - Solution: Retrain with recent data, add drift detection
        
        DIAGNOSTIC CHECKLIST:
        
        □ Base model performances documented
        □ Ensemble improvement >3% over best base
        □ Prediction correlations <0.9 between base models
        □ Meta-learner weights reasonably balanced
        □ Cross-validation strategy appropriate for data structure
        □ No time-series or data leakage detected
        □ Error analysis by key segments completed
        □ Training-validation gap <10%
        □ Feature importance stable across CV folds
        □ Business validation completed
        □ Predictions within reasonable ranges
        □ Model complexity justified by performance gain
        
        DECISION MATRIX:
        
        Ensemble RMSE 35, Best Base RMSE 36:
        - Improvement: 2.9%
        - Decision: Not worth ensemble complexity, use best base model
        
        Ensemble RMSE 32, Best Base RMSE 36:
        - Improvement: 11.1%
        - Decision: Deploy ensemble if complexity manageable
        
        Ensemble RMSE 37, Best Base RMSE 36:
        - Improvement: -2.8% (worse!)
        - Decision: Debug ensemble, likely overfitting or lack of diversity
        """,
        "metadata": {
            "diagnostic_steps": [
                "baseline_comparison",
                "diversity_check",
                "meta_learner_weights",
                "data_split_validation",
                "error_pattern_analysis",
                "overfitting_detection",
                "feature_importance_stability",
                "business_validation"
            ],
            "correlation_threshold": 0.9,
            "minimum_improvement": "3-5%",
            "overfitting_gap_threshold": 0.10
        },
        "keywords": ["troubleshooting", "debugging", "performance issues", "overfitting", "model diagnostics", "error analysis"]
    },
    
    {
        "doc_id": "DEBUG002",
        "category": "troubleshooting",
        "title": "Explaining Why Ensemble Outperforms or Underperforms",
        "content": """
        Framework for generating narrative explanations of ensemble performance 
        relative to base models for business stakeholders.
        
        ENSEMBLE OUTPERFORMANCE EXPLANATIONS:
        
        Scenario 1: Complementary Strengths
        
        Data pattern:
        - Random Forest RMSE: 40 (good at capturing non-linear patterns)
        - XGBoost RMSE: 38 (excellent at sequential patterns)
        - Linear Regression RMSE: 50 (captures linear trends)
        - Ensemble RMSE: 32
        
        Explanation:
        "The ensemble achieves 16% better accuracy than the best individual model 
        (XGBoost) by combining complementary strengths:
        
        - XGBoost excels at capturing prescription trends over time
        - Random Forest is more robust to outlier HCPs and unusual behaviors
        - Linear Regression provides stable baseline predictions
        
        The meta-learner learns that:
        - For high-volume HCPs: Trust XGBoost (weight 0.60)
        - For volatile HCPs: Trust Random Forest (weight 0.55)
        - For new HCPs with limited history: Trust Linear Model (weight 0.50)
        
        By selecting the right model for each prediction context, the ensemble 
        reduces both bias (systematic errors) and variance (prediction instability)."
        
        Scenario 2: Error Averaging
        
        Data pattern:
        - All base models RMSE 40-42
        - Base model errors uncorrelated
        - Ensemble RMSE: 36
        
        Explanation:
        "Each base model makes different errors on different HCPs. When we average 
        their predictions, random errors cancel out while systematic patterns reinforce.
        
        Example HCP:
        - Actual NRx: 50
        - Random Forest predicts: 45 (error: -5)
        - XGBoost predicts: 55 (error: +5)
        - LightGBM predicts: 48 (error: -2)
        - Ensemble average: 49.3 (error: -0.7)
        
        The ensemble reduces error by 86% for this HCP through intelligent averaging."
        
        Scenario 3: Interaction Coverage
        
        Data pattern:
        - Simple models miss complex interactions
        - Ensemble captures specialty × marketing × history interactions
        
        Explanation:
        "HCP prescribing behavior involves complex interactions between specialty, 
        marketing touchpoints, and historical prescribing patterns. No single model 
        type captures all interaction patterns:
        
        - Trees capture: Specialty-specific call response curves
        - Boosting captures: Marketing momentum effects over time
        - Linear model captures: Overall market trends
        
        The ensemble combines these complementary interaction patterns, improving 
        predictions especially for:
        - Primary care HCPs with high sample exposure (+12% accuracy)
        - Cardiologists with recent marketing intensity (+15% accuracy)
        - New prescribers with limited history (+20% accuracy)"
        
        ENSEMBLE UNDERPERFORMANCE EXPLANATIONS:
        
        Scenario 1: Lack of Diversity
        
        Data pattern:
        - All base models very similar (correlation >0.95)
        - Ensemble RMSE = 40, Best base RMSE = 40
        
        Explanation:
        "The ensemble shows minimal improvement over individual models because all 
        base models (XGBoost, LightGBM, CatBoost) are variants of gradient boosting. 
        They make the same errors on the same HCPs.
        
        Prediction correlations:
        - XGBoost vs LightGBM: 0.97
        - XGBoost vs CatBoost: 0.96
        - LightGBM vs CatBoost: 0.98
        
        With such high correlation, averaging provides no benefit. The ensemble adds 
        complexity without improving accuracy.
        
        Recommendation: Replace two models with different algorithm types (Random Forest, 
        Linear Regression, or Neural Network) to increase diversity."
        
        Scenario 2: One Dominant Model
        
        Data pattern:
        - XGBoost RMSE: 35
        - Random Forest RMSE: 60
        - Linear Regression RMSE: 75
        - Ensemble RMSE: 38
        
        Explanation:
        "The ensemble performs worse than using XGBoost alone because the weak base 
        models (Random Forest and Linear Regression) add noise rather than signal.
        
        Meta-learner learned weights:
        - XGBoost: 0.82
        - Random Forest: 0.12
        - Linear Regression: 0.06
        
        The weak models contribute 18% to final predictions but have error rates 2x 
        higher than XGBoost, degrading overall performance.
        
        Recommendation: Remove weak models or improve them to within 20% of XGBoost 
        performance before including in ensemble."
        
        Scenario 3: Overfitting
        
        Data pattern:
        - Validation R²: 0.85
        - Test R²: 0.68
        - Gap: 0.17
        
        Explanation:
        "The ensemble overfit to the validation set, learning patterns that don't 
        generalize to new data.
        
        Contributing factors:
        1. Complex meta-learner (neural network with 3 hidden layers)
        2. Small validation set (2,000 HCPs)
        3. Many base models (7) relative to validation size
        
        The meta-learner learned to exploit validation set quirks rather than true 
        predictive patterns.
        
        Evidence:
        - Meta-learner weights vary wildly across CV folds
        - Top features different in each fold
        - Performance cliff when deployed to production
        
        Recommendation: Use simpler meta-learner (Ridge Regression), increase validation 
        set size, or reduce number of base models to 3-4."
        
        Scenario 4: Insufficient Data
        
        Data pattern:
        - Training sample: 5,000 HCPs
        - 5 base models + meta-learner
        - High variance across CV folds
        
        Explanation:
        "The ensemble has insufficient data to reliably train multiple models plus a 
        meta-learner.
        
        Cross-validation results:
        - Fold 1 R²: 0.72
        - Fold 2 R²: 0.58
        - Fold 3 R²: 0.81
        - Fold 4 R²: 0.64
        - Standard deviation: 0.10 (high!)
        
        With only 1,000 HCPs per fold, we can't robustly estimate model parameters. 
        Each fold gives different feature importance rankings and meta-learner weights.
        
        Rule of thumb: Need 10,000+ observations for stable ensemble training.
        
        Recommendation: Use single well-tuned model (XGBoost or Random Forest) until 
        more data collected, or simplify ensemble to 2-3 base models."
        
        NARRATIVE TEMPLATE FOR BUSINESS STAKEHOLDERS:
        
        "The [ensemble/base model] achieved [metric value] on [use case], representing 
        a [X%] [improvement/degradation] compared to [baseline].
        
        This performance difference is primarily due to [root cause]:
        
        [Specific technical explanation in business terms]
        
        Key supporting evidence:
        - [Quantitative finding 1]
        - [Quantitative finding 2]
        - [Quantitative finding 3]
        
        Business impact:
        [Translation to business outcomes - prescriptions, revenue, targeting efficiency]
        
        Recommendation:
        [Clear action - deploy, iterate, or abandon ensemble]
        
        Confidence level: [High/Medium/Low] based on [validation approach]"
        
        EXAMPLE COMPLETE EXPLANATION:
        
        "The NRx forecasting ensemble achieved RMSE of 32 prescriptions/month, 
        representing an 11% improvement over our best individual model (XGBoost at 36 RMSE).
        
        This improvement is primarily due to complementary model strengths combined with 
        intelligent context-aware weighting:
        
        XGBoost excels at identifying prescription trends and momentum but struggles with 
        volatile HCPs who have irregular prescribing patterns. Random Forest is more 
        robust to these outliers but less accurate for stable trendlines. The ensemble 
        meta-learner learns to trust XGBoost for stable HCPs (60% of portfolio) and 
        Random Forest for volatile HCPs (40% of portfolio).
        
        Key supporting evidence:
        - Base model prediction correlation: 0.76 (good diversity)
        - Meta-learner weights balanced: 0.55 XGBoost, 0.45 Random Forest
        - Improvement consistent across 5 CV folds (RMSE 31-33)
        - Test set performance matches validation (no overfitting)
        
        Business impact:
        11% forecast accuracy improvement translates to:
        - $2.3M reduction in forecast error (vs actual prescriptions)
        - 15% improvement in territory allocation efficiency
        - Higher confidence in quarterly guidance to senior leadership
        
        Recommendation:
        Deploy ensemble to production for Q1 2025 forecasting cycle. Continue monitoring 
        individual base model performance monthly.
        
        Confidence level: High - validated across multiple time periods, geographies, 
        and HCP segments with consistent improvement."
        """,
        "metadata": {
            "explanation_types": [
                "complementary_strengths",
                "error_averaging",
                "interaction_coverage",
                "lack_of_diversity",
                "one_dominant_model",
                "overfitting",
                "insufficient_data"
            ],
            "narrative_components": [
                "performance_summary",
                "root_cause",
                "technical_explanation",
                "supporting_evidence",
                "business_impact",
                "recommendation",
                "confidence_level"
            ]
        },
        "keywords": ["performance explanation", "ensemble advantage", "root cause analysis", "business communication", "narrative generation"]
    }
]


# ========================================================================
# HELPER FUNCTIONS FOR VECTOR DB INGESTION
# ========================================================================

def get_documents_by_category(category: str) -> list:
    """
    Retrieve all documents for a specific category.
    
    Args:
        category: One of 'use_case', 'ensemble_method', 'metric', 
                  'business_context', 'features', 'troubleshooting'
    
    Returns:
        List of document dictionaries
    """
    return [doc for doc in VECTOR_DB_DOCUMENTS if doc['category'] == category]


def get_document_by_id(doc_id: str) -> dict:
    """
    Retrieve a specific document by ID.
    
    Args:
        doc_id: Document identifier (e.g., 'UC001', 'ENS001')
    
    Returns:
        Document dictionary or None if not found
    """
    for doc in VECTOR_DB_DOCUMENTS:
        if doc['doc_id'] == doc_id:
            return doc
    return None


def get_all_keywords() -> set:
    """
    Extract all unique keywords across all documents.
    
    Returns:
        Set of all keywords
    """
    keywords = set()
    for doc in VECTOR_DB_DOCUMENTS:
        keywords.update(doc['keywords'])
    return keywords


def search_by_keyword(keyword: str) -> list:
    """
    Find all documents containing a specific keyword.
    
    Args:
        keyword: Search term
    
    Returns:
        List of matching documents
    """
    keyword_lower = keyword.lower()
    matching_docs = []
    
    for doc in VECTOR_DB_DOCUMENTS:
        if any(keyword_lower in kw.lower() for kw in doc['keywords']):
            matching_docs.append(doc)
    
    return matching_docs


# ========================================================================
# EXPORT SUMMARY
# ========================================================================

DOCUMENT_SUMMARY = {
    "total_documents": len(VECTOR_DB_DOCUMENTS),
    "categories": {
        "use_case": len(get_documents_by_category("use_case")),
        "ensemble_method": len(get_documents_by_category("ensemble_method")),
        "metric": len(get_documents_by_category("metric")),
        "business_context": len(get_documents_by_category("business_context")),
        "features": len(get_documents_by_category("features")),
        "troubleshooting": len(get_documents_by_category("troubleshooting"))
    },
    "total_keywords": len(get_all_keywords()),
    "coverage": [
        "5 pharma use cases (NRx, HCP response, feature importance, drift, messaging)",
        "5 ensemble methods (fundamentals, boosting, bagging, stacking, blending)",
        "3 metric guides (regression, classification, uplift)",
        "3 business context docs (lifecycle, terminology, failure modes)",
        "2 feature guides (HCP features, interactions)",
        "2 troubleshooting guides (diagnostics, explanations)"
    ]
}