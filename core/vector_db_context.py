- ROI: Estimated 2-3x over 12 months
        - Decision: Implement
        
        ## Feature Importance Changes Over Time
        
        **Why Rankings Change**:
        - Market dynamics shift (competitor launches)
        - Product lifecycle stage (launch vs mature)
        - Seasonal patterns (flu season changes drivers)
        - Strategic shifts (increased promotional intensity)
        - External events (guideline updates, safety warnings)
        
        **What to Monitor**:
        - Major rank changes (>5 positions) signal market shifts
        - New features entering top 10: Emerging patterns
        - Previously important features dropping: Declining relevance
        - Importance score changes: Strength of relationship changing
        
        **Example Drift Signal**:
        "competitor_market_share" jumped from rank 12 to rank 3
        → Diagnosis: Competitive pressure increased dramatically
        → Investigation: Check for new competitor launches, aggressive campaigns
        → Action: Respond with competitive messaging, increase share-of-voice
        
        ## Common Mistakes in Feature Interpretation
        
        **Mistake 1: Confusing Prediction with Causation**
        - Feature important ≠ Feature causes outcome
        - Example: "hcp_age" may be important because it correlates with specialty
        - Fix: Use domain knowledge to assess causal plausibility
        
        **Mistake 2: Ignoring Correlated Features**
        - If call_frequency and email_frequency are highly correlated (0.9+)
        - Model may arbitrarily choose one as "important"
        - Reality: Both contribute, but model can't separate them
        - Fix: Look at combined "promotional_intensity" rather than individual channels
        
        **Mistake 3: Focusing Only on Top Feature**
        - Top feature is important, but not the whole story
        - Multiple features work together
        - Fix: Consider top 10-20 features, look for interactions
        
        **Mistake 4: Treating All Importance Types as Equal**
        - Gain importance ≠ Permutation importance ≠ SHAP
        - Each measures different aspect
        - Fix: Use multiple importance methods, triangulate insights
        
        **Mistake 5: Ignoring Non-Actionable Features**
        - Spending time on features you can't control
        - Example: Deep analysis of "hcp_years_in_practice" (can't change it)
        - Fix: Focus analysis on actionable features for business recommendations
        """,
        "tags": ["feature_importance", "drivers", "interpretability", "actionable_insights", "SHAP", "business_translation"],
        "use_cases": ["feature_importance_analysis", "NRx_forecasting", "HCP_engagement", "messaging_optimization"]
    },
    
    # Document 5: Model Drift and Performance Monitoring
    {
        "id": "domain_model_drift_001",
        "category": "domain_knowledge",
        "subcategory": "model_monitoring",
        "title": "Model Drift Detection and Performance Degradation",
        "content": """
        Model drift occurs when a model's performance degrades over time due to changes in
        underlying patterns. In pharmaceutical commercial analytics, markets evolve rapidly,
        making drift detection critical for maintaining prediction accuracy.
        
        ## Why Models Drift in Pharma
        
        **Markets Are Dynamic**:
        - Competitor launches change prescribing landscape (every 6-12 months)
        - Generic entry causes catastrophic brand erosion (overnight changes)
        - Clinical guidelines update (annual or ad-hoc)
        - Safety warnings emerge (sudden, unpredictable)
        - Formulary changes (quarterly or annual)
        - HCP demographics shift (annual residency cohorts)
        
        **Models Are Static**:
        - Trained on historical patterns (6-24 months of data)
        - Assume relationships remain constant
        - Can't adapt automatically to new patterns
        - Require periodic retraining to stay current
        
        ## Types of Drift
        
        **1. Concept Drift** (Relationship Changes):
        
        Definition: The relationship between features (X) and target (Y) changes
        
        Pharma Examples:
        - New competitor launch → promotional effectiveness decreases
        - Generic entry → brand prescribing plummets regardless of promotion
        - Guideline change → specialty prescribing patterns shift
        - Formulary restriction → price sensitivity increases
        
        Detection Signals:
        - Feature distributions stable (PSI < 0.1)
        - BUT model performance degrades significantly
        - Feature importance rankings change dramatically
        - Coefficients/patterns in model differ from baseline
        
        Business Impact:
        "Model says call this HCP 4 times, but now they're non-responsive due to competitor"
        → Wasted promotional spend, missed opportunities elsewhere
        
        **2. Data Drift** (Feature Distribution Changes):
        
        Definition: The distribution of input features (X) changes
        
        Pharma Examples:
        - HCP demographics shift (younger physicians entering, older retiring)
        - Territory realignment changes geographic distribution
        - Sales force reorganization alters call frequency patterns
        - Data quality issues (system migration, missing values)
        
        Detection Signals:
        - PSI (Population Stability Index) > 0.25 for key features
        - Feature means, standard deviations shift significantly
        - New categories appear (new specialties, geographies)
        - Missing values increase
        
        Business Impact:
        "Model trained on experienced HCPs, now many are new graduates who prescribe differently"
        → Predictions systematically biased for new demographic segments
        
        **3. Performance Drift** (Accuracy Decline):
        
        Definition: Model accuracy degrades over time
        
        Pharma Examples:
        - RMSE increases from 10.0 to 12.5 (25% degradation)
        - R² decreases from 0.75 to 0.65
        - Prediction errors grow consistently over quarters
        
        Usually caused by combination of concept drift + data drift
        
        **4. Prediction Drift** (Output Distribution Changes):
        
        Definition: Distribution of model predictions shifts
        
        Pharma Examples:
        - Mean predicted NRx drops from 50 to 40 (but actuals unchanged)
        - Prediction variance increases (model less confident)
        - Model consistently over-predicts or under-predicts
        
        Detection Signals:
        - Track prediction distribution statistics over time
        - Compare predicted vs actual distributions
        - Monitor calibration (predicted probabilities match observed rates)
        
        ## Drift Severity Classification
        
        **Minor Drift** (5-10% performance degradation):
        - Impact: Slightly reduced accuracy, still usable
        - Action: Monitor closely, schedule retraining in 1-2 months
        - Business risk: Low, tactical adjustments can compensate
        - Example: RMSE 10.0 → 10.7
        
        **Moderate Drift** (10-20% performance degradation):
        - Impact: Noticeably worse predictions, targeting suboptimal
        - Action: Plan retraining within 2-4 weeks
        - Business risk: Medium, some wasted spend and missed opportunities
        - Example: RMSE 10.0 → 11.5
        
        **Severe Drift** (>20% performance degradation):
        - Impact: Model unreliable, predictions misleading
        - Action: Immediate emergency retraining required
        - Business risk: High, significant revenue at stake
        - Example: RMSE 10.0 → 13.0
        - Business cost: Can represent $5-15M annual forecast error
        
        ## Root Cause Analysis Framework
        
        **Step 1: Quantify Degradation**
        - Compare current performance to baseline (6-12 months ago)
        - Calculate percentage change in key metrics (RMSE, MAE, R²)
        - Identify when degradation started (gradual or sudden?)
        
        **Step 2: Check for Concept Drift**
        - Compare feature importance: current vs baseline
        - Analyze residual patterns (systematic over/under prediction?)
        - Review feature distributions (stable = concept drift likely)
        - Check external events (competitor launches, guidelines, formulary)
        
        **Step 3: Check for Data Drift**
        - Calculate PSI for each feature
        - Identify features with PSI > 0.25 (significant shift)
        - Investigate: Why did distribution change?
        - Data quality: Missing values, system changes?
        
        **Step 4: Segment Analysis**
        - Does drift affect all HCP tiers equally?
        - Is drift concentrated in specific geographies?
        - Which specialties show most degradation?
        - Urban vs rural differences?
        
        This reveals: Is drift market-wide or segment-specific?
        
        **Step 5: External Event Investigation**
        - Review market timeline: What happened in drift period?
        - Competitor activity: Launches, campaigns, pricing
        - Regulatory: Formulary updates, prior auth changes
        - Clinical: New studies, safety warnings, guideline updates
        - Operational: Sales force changes, territory realignment
        
        ## Business Impact Quantification
        
        **Framework**:
        Forecast Error Impact = (RMSE_increase) × (# HCPs) × (Revenue per Rx) × (Periods per year)
        
        **Example Calculation**:
        - RMSE increased from 10 to 12.5 NRx (+2.5)
        - Forecasting 10,000 HCPs monthly
        - Revenue: $50 per Rx
        - Impact: 2.5 NRx × 10,000 HCPs × $50 × 12 months = $15M annual error
        
        **Operational Impacts**:
        - Inventory: Over-forecast → waste, under-forecast → stockouts
        - Sales force: Mis-targeted HCPs → wasted calls, missed opportunities
        - Marketing: Budget allocated to wrong campaigns/channels
        - Planning: Unrealistic quotas, poor territory alignment
        
        ## Remediation Strategies
        
        **Quick Fixes** (1-2 weeks):
        - Recalibration: Adjust prediction scale without full retraining
        - Feature updates: Add recent competitor activity data
        - Segment models: Separate models for drifted vs stable segments
        - Ensemble reweighting: If using ensemble, adjust base model weights
        
        **Full Retraining** (2-4 weeks):
        - Expand training data: Include recent periods showing new patterns
        - Feature engineering: Add features capturing market changes
        - Algorithm change: Switch to more adaptive algorithm if needed
        - Validation: Extensive testing on recent data before deployment
        
        **Architectural Changes** (1-3 months):
        - Online learning: Model updates continuously with new data
        - Drift detection system: Automated monitoring and alerts
        - Concept drift algorithms: Models designed to adapt to change
        - Modular approach: Separate models for stable vs volatile segments
        
        ## Prevention Best Practices
        
        **Proactive Monitoring**:
        - Weekly performance checks (RMSE, MAE, R²)
        - Monthly drift analysis (PSI, feature importance)
        - Quarterly retraining schedule (don't wait for drift)
        - Automated alerts (10% degradation threshold)
        
        **Business Event Tracking**:
        - Maintain competitor launch calendar
        - Track formulary change schedule
        - Monitor clinical guideline updates
        - Log safety warnings, regulatory changes
        - Correlate events with model performance
        
        **Segment-Level Monitoring**:
        - Don't just track overall performance
        - Monitor by HCP tier, specialty, geography
        - Detect segment-specific drift early
        - Enables targeted remediation
        
        **Model Versioning**:
        - Keep historical model versions
        - Track performance over time
        - Enable rollback if new model worse
        - A/B test new vs old model before full deployment
        
        ## Retraining Decision Framework
        
        **When to Retrain**:
        
        Time-based triggers:
        - Quarterly retraining (proactive, prevents drift)
        - After major market events (competitor launch, generic entry)
        - Annual comprehensive refresh
        
        Performance-based triggers:
        - 10% degradation: Plan retraining within month
        - 15% degradation: Urgent retraining within 2 weeks
        - 20%+ degradation: Emergency retraining immediately
        
        Business event triggers:
        - Major competitor launch: Retrain within 1 month
        - Generic entry: Retrain immediately (market transformed)
        - Guideline change: Retrain within 2 months
        - Sales force reorganization: Retrain after stabilization (2-3 months)
        
        **Cost-Benefit Analysis**:
        
        Retraining costs:
        - Data science time: 40-120 hours ($6K-$18K)
        - Compute resources: $2K-$10K
        - Testing and validation: $3K-$8K
        - Deployment: $2K-$5K
        - Total: $13K-$41K typical
        
        Benefits of fixing 20% degradation:
        - Restore forecast accuracy: $10M-$20M impact
        - Conservative capture: 50% of potential = $5M-$10M
        - ROI: 200-700x
        
        Decision: Almost always retrain when degradation >10%
        
        ## Communicating Drift to Stakeholders
        
        **Framework**:
        1. Quantify performance change (numbers stakeholders understand)
        2. Explain root cause in business terms (not technical jargon)
        3. Estimate business impact (revenue, costs, missed opportunities)
        4. Present remediation plan with timeline
        5. Show ROI of fixing vs ignoring
        6. Request resources/approval if needed
        
        **Example Message**:
        "Our NRx forecasting model has experienced 18% accuracy degradation over the past
        3 months (RMSE increased from 10 to 11.8). Root cause analysis indicates this is
        due to the competitor launch in March, which changed the relationship between our
        promotional activities and HCP prescribing behavior.
        
        Business impact: $8M annual forecast error, leading to suboptimal targeting and
        wasted promotional spend estimated at $2M.
        
        Recommended action: Emergency retraining within 2 weeks, incorporating competitor
        activity features and recent data showing new market dynamics. Cost: $25K.
        Expected benefit: Restore 75% of lost accuracy = $6M annual value. ROI: 240x.
        
        Request: Approval for immediate retraining and resources (80 hours data science time)."
        """,
        "tags": ["model_drift", "monitoring", "performance_degradation", "concept_drift", "data_drift", "retraining", "PSI"],
        "use_cases": ["model_drift_detection", "NRx_forecasting", "HCP_engagement"]
    },
    
    # Document 6: Ensemble Methods
    {
        "id": "domain_ensemble_methods_001",
        "category": "domain_knowledge",
        "subcategory": "ensemble_modeling",
        "title": "Ensemble Methods and Model Combination Strategies",
        "content": """
        Ensemble models combine predictions from multiple base models to achieve better
        performance than any individual model. In pharmaceutical analytics, ensembles are
        particularly valuable due to complex, noisy data and high business stakes.
        
        ## Why Ensembles Work
        
        **The Wisdom of Crowds Principle**:
        - Multiple models make different errors
        - Averaging reduces random error (variance)
        - Combination can reduce systematic error (bias)
        - Result: More stable, accurate predictions
        
        **The Diversity Requirement**:
        - Benefit is proportional to model diversity
        - If all models identical → no ensemble advantage
        - If models very different → maximum ensemble benefit
        - Goal: Diverse algorithms, features, or training approaches
        
        ## Ensemble Types
        
        **1. Bagging (Bootstrap Aggregating)**:
        
        Mechanism:
        - Train multiple models on different random samples of data
        - Each model sees slightly different view of patterns
        - Average predictions across all models
        
        Example: Random Forest (ensemble of decision trees)
        
        Strengths:
        - Reduces variance (smooths out overfitting)
        - Robust to outliers
        - Parallel training (fast)
        
        Pharma Application:
        - NRx forecasting: Multiple models on different HCP subsamples
        - Handles noisy data well (common in pharma)
        - Good for high-variance base models
        
        Typical Improvement: 10-20% RMSE reduction vs single tree
        
        **2. Boosting**:
        
        Mechanism:
        - Train models sequentially
        - Each new model focuses on errors of previous models
        - Weighted combination (better models get more weight)
        
        Examples: XGBoost, LightGBM, AdaBoost
        
        Strengths:
        - Reduces bias (captures subtle patterns)
        - Excellent predictive performance
        - Handles complex interactions
        
        Pharma Application:
        - HCP engagement prediction: Complex interaction patterns
        - Feature importance analysis: Identifies subtle drivers
        - Best for achieving maximum accuracy
        
        Typical Improvement: 15-30% RMSE reduction vs linear models
        
        **3. Stacking (Stacked Generalization)**:
        
        Mechanism:
        - Train diverse base models (Random Forest, XGBoost, Logistic Regression)
        - Use base model predictions as features for meta-learner
        - Meta-learner learns optimal combination
        
        Example Architecture:
        - Layer 1: Random Forest + XGBoost + Neural Network
        - Layer 2: Linear Regression or Ridge Regression (meta-learner)
        
        Strengths:
        - Combines strengths of different algorithm types
        - Meta-learner optimizes weights automatically
        - Maximum potential for improvement
        
        Pharma Application:
        - High-stakes forecasting (launch forecasts)
        - When maximum accuracy justifies complexity
        - Combines interpretable + black-box models
        
        Typical Improvement: 5-15% beyond best base model
        
        **4. Blending**:
        
        Mechanism:
        - Similar to stacking but simpler
        - Train base models on training data
        - Train meta-learner on holdout validation set
        - Less prone to overfitting than stacking
        
        Pharma Application:
        - When training data limited (small therapeutic areas)
        - Simpler alternative to full stacking
        - Good for production deployment (less complex)
        
        **5. Voting/Averaging**:
        
        Mechanism:
        - Simple average (regression) or majority vote (classification)
        - No meta-learner, just combine predictions directly
        - Can use weighted average based on validation performance
        
        Pharma Application:
        - Quick ensemble for proof-of-concept
        - When interpretability matters (simple to explain)
        - Baseline ensemble approach
        
        Typical Improvement: 3-8% vs best base model
        
        ## Designing Effective Ensembles
        
        **Base Model Selection**:
        
        Goal: Maximum diversity while maintaining quality
        
        Good Combinations:
        - Linear + Tree-based + Neural Network
          (Different inductive biases)
        - Random Forest + XGBoost + LightGBM
          (Different tree-building strategies)
        - Logistic Regression + SVM + Random Forest
          (Linear + nonlinear combinations)
        
        Poor Combinations:
        - Multiple Random Forests with similar hyperparameters (too similar)
        - Multiple linear models (all capture same patterns)
        - Many weak models (garbage in, garbage out)
        
        **Number of Base Models**:
        - 2-3 models: Good starting point, easy to explain
        - 4-5 models: Typical sweet spot (diminishing returns after)
        - 10+ models: Usually marginal improvement, high complexity
        
        Rule of thumb: Stop adding models when improvement <2%
        
        **Meta-Learner Choice**:
        
        For stacking ensembles:
        - Simple meta-learner (Linear/Ridge Regression): Prevents overfitting
        - Complex meta-learner (Neural Network): Risks overfitting to base predictions
        
        Recommendation: Start simple, only increase complexity if needed
        
        ## Ensemble Performance Analysis
        
        **Comparing Ensemble to Base Models**:
        
        Metrics to Compare:
        - RMSE/MAE: Is ensemble more accurate?
        - R²: Does ensemble explain more variance?
        - Improvement %: (Base_RMSE - Ensemble_RMSE) / Base_RMSE × 100
        - Consistency: Is ensemble better across all segments?
        
        **Example Analysis**:
        - Random Forest: RMSE = 10.5
        - XGBoost: RMSE = 10.2
        - LightGBM: RMSE = 10.8
        - Average Base: RMSE = 10.5
        - Ensemble: RMSE = 9.8
        - Improvement: (10.5 - 9.8) / 10.5 = 6.7%
        
        Questions to Answer:
        1. Is 6.7% improvement significant? (Usually yes if >5%)
        2. Does improvement justify added complexity? (Cost-benefit analysis)
        3. Is improvement consistent across HCP tiers? (Check segments)
        4. Which base models contribute most? (Analyze weights/importance)
        
        ## When Ensembles Underperform
        
        **Warning Signs**:
        
        1. **High Base Model Correlation** (>0.95):
        - Models make same errors
        - Little diversity benefit
        - Fix: Add more diverse algorithms or features
        
        2. **Overfitting Meta-Learner**:
        - Ensemble great on validation, poor on test
        - Meta-learner memorized base predictions
        - Fix: Use simpler meta-learner (Linear Regression), add regularization
        
        3. **Poor Base Model Selection**:
        - Including very weak models (much worse than others)
        - Weak models drag down ensemble
        - Fix: Pre-screen base models, only include strong performers
        
        4. **Data Leakage**:
        - Base models trained on same data as meta-learner
        - Artificially inflated performance
        - Fix: Strict train/validation/test splits, use cross-validation
        
        5. **Model Saturation**:
        - Already at theoretical performance limit
        - No room for improvement
        - Fix: Accept that ensemble can't help, focus on feature engineering
        
        ## Business Value Assessment
        
        **Cost of Ensemble Complexity**:
        - Training time: 3-5x longer than single model
        - Compute resources: 2-4x higher costs
        - Maintenance: More models to monitor and retrain
        - Explainability: Harder to interpret (black box)
        - Deployment: More complex infrastructure
        
        Annual Complexity Cost: $5K-$15K typically
        
        **Benefit Calculation**:
        
        Example: NRx Forecasting Ensemble
        - 6.7% RMSE improvement (10.5 → 9.8)
        - 10,000 HCPs forecasted monthly
        - Error reduction: 0.7 NRx per HCP per month
        - Annual: 0.7 × 10,000 × 12 = 84,000 fewer prescription errors
        - Value: 84,000 × $50 = $4.2M
        - Complexity cost: $10K
        - ROI: 420x
        
        Decision: Strongly justified
        
        **When to Use Ensemble**:
        - High-stakes decisions ($10M+ revenue impact)
        - Accuracy improvement >5% vs best base model
        - Diminishing returns not yet reached
        - Resources available for complexity
        
        **When to Use Single Model**:
        - Exploratory analysis (speed matters)
        - Interpretability critical (stakeholder requirement)
        - Improvement <3% (not worth complexity)
        - Limited computational resources
        - Proof-of-concept phase
        
        ## Ensemble-Specific Insights
        
        **Feature Importance from Ensembles**:
        
        Ensembles can reveal patterns missed by individual models:
        - Feature interactions: Ensemble identifies synergies
        - Robust drivers: Features important across all base models
        - Ensemble-unique features: Meta-learner discovers new patterns
        
        Example:
        "Random Forest ranks 'patient_volume' as #5, XGBoost ranks it #8, but ensemble
        ranks it #3. This suggests the interaction between patient_volume and other features
        is critical, which the ensemble meta-learner discovered."
        
        **Segment Performance**:
        
        Check if ensemble advantage consistent:
        - TIER_1 HCPs: Ensemble RMSE 8.5 vs Base 9.2 (7.6% improvement)
        - TIER_2 HCPs: Ensemble RMSE 9.8 vs Base 10.5 (6.7% improvement)
        - TIER_3 HCPs: Ensemble RMSE 11.2 vs Base 11.5 (2.6% improvement)
        
        Insight: Ensemble especially valuable for high-volume HCPs (where accuracy matters most)
        
        ## Communicating Ensemble Value
        
        **To Technical Stakeholders**:
        - Show performance metrics (RMSE, R², improvement %)
        - Explain algorithmic diversity and combination strategy
        - Present ablation analysis (removing each base model's impact)
        - Discuss bias-variance tradeoff
        
        **To Business Stakeholders**:
        - Translate accuracy improvement to revenue impact
        - Use analogy: "Like getting second and third opinions from doctors"
        - Emphasize risk reduction (more stable predictions)
        - Show ROI calculation (benefit vs complexity cost)
        - Avoid technical jargon (no mention of meta-learners, stacking)
        
        **Example Message**:
        "The ensemble model combines three complementary prediction approaches: Random Forest
        (handles outliers well), XGBoost (captures subtle patterns), and Linear Regression
        (provides interpretable baseline). By learning from all three, the ensemble achieves
        6.7% better accuracy than any single approach.
        
        Business impact: This improvement translates to $4.2M better forecast accuracy annually,
        enabling more precise HCP targeting and resource allocation. The additional complexity
        costs $10K annually but delivers 420x ROI. Recommend deployment to production."
        """,
        "tags": ["ensemble", "stacking", "boosting", "bagging", "model_combination", "accuracy", "complexity"],
        "use_cases": ["NRx_forecasting", "HCP_engagement", "feature_importance_analysis"]
    },
    
    # Document 7: Uplift Modeling
    {
        "id": "domain_uplift_modeling_001",
        "category": "domain_knowledge",
        "subcategory": "uplift_modeling",
        "title": "Uplift Modeling and Incremental Impact Measurement",
        "content": """
        Uplift modeling predicts the INCREMENTAL impact of an intervention (promotion, campaign)
        on an individual's behavior. Unlike traditional response modeling which predicts WHO will
        respond, uplift modeling predicts who will respond BECAUSE of the intervention.
        
        This is the difference between correlation and causation in targeting.
        
        ## The Core Problem
        
        **Traditional Response Modeling Question**: "Will this HCP increase prescriptions?"
        
        Problem: Some HCPs would increase anyway (without promotion)
        → Wastes resources on "sure things"
        
        **Uplift Modeling Question**: "Will this HCP increase prescriptions MORE if promoted?"
        
        Focus: Incremental lift from promotion
        → Targets only HCPs influenced by promotion
        
        ## The Four Customer Segments
        
        **1. Persuadables** (20-30% typically):
        - Will respond ONLY if treated
        - Positive uplift: Treatment causes behavior change
        - **TARGET THESE** → Maximum ROI
        
        Example HCP:
        - Currently prescribes 10 NRx/month
        - If promoted: 25 NRx/month
        - If not promoted: 10 NRx/month
        - Uplift: +15 NRx (worth targeting!)
        
        **2. Sure Things** (10-20% typically):
        - Will respond regardless of treatment
        - Zero uplift: Would prescribe anyway
        - **AVOID** → Wasted promotion spend
        
        Example HCP:
        - Currently prescribes 50 NRx/month
        - If promoted: 55 NRx/month
        - If not promoted: 53 NRx/month
        - Uplift: +2 NRx (not worth $200 call cost)
        
        **3. Lost Causes** (40-50% typically):
        - Won't respond regardless of treatment
        - Zero uplift: Unpersuadable
        - **AVOID** → Wasted effort
        
        Example HCP:
        - Currently prescribes 0 NRx/month
        - If promoted: 0 NRx/month
        - If not promoted: 0 NRx/month
        - Uplift: 0 (competitor loyalist, formulary blocker, etc.)
        
        **4. Do Not Disturbs/Sleeping Dogs** (<5% typically):
        - Negative response if treated
        - Negative uplift: Promotion backfires
        - **DEFINITELY AVOID** → Harmful to promote
        
        Example HCP:
        - Currently prescribes 10 NRx/month
        - If promoted: 3 NRx/month (promotion annoys them)
        - If not promoted: 10 NRx/month
        - Uplift: -7 NRx (promotion causes opt-out)
        
        ## Why Uplift Modeling Matters
        
        **Traditional Targeting Results**:
        - Email 10,000 HCPs
        - 2,000 respond (20% response rate)
        - Cost: $5,000
        - Revenue: 2,000 × 5 NRx × $50 = $500K
        - ROI: 100x (looks great!)
        
        **But Reality Check** (with uplift lens):
        - 400 are persuadables (truly influenced)
        - 1,600 are "sure things" (would prescribe anyway)
        - True incremental: 400 × 5 NRx × $50 = $100K
        - Actual ROI: 20x (still good, but 80% less than calculated)
        
        **Uplift-Optimized Targeting**:
        - Target only 2,000 persuadables (from uplift model)
        - 800 respond (40% response rate among persuadables)
        - Cost: $1,000 (80% savings)
        - Revenue: 800 × 5 NRx × $50 = $200K (100% incremental)
        - ROI: 200x (truly measured incremental impact)
        
        **Key Insight**: Uplift modeling can double or triple marketing ROI by avoiding waste
        on sure things and lost causes.
        
        ## Measurement Challenges
        
        **The Fundamental Problem**:
        - Can only observe ONE outcome per HCP (promoted or not promoted)
        - Cannot see counterfactual (what would have happened otherwise)
        - Need to estimate incremental effect from observed data
        
        **Gold Standard: Randomized Controlled Trial (RCT)**:
        - Randomly assign HCPs to treatment vs control
        - Treatment: Receive promotion
        - Control: No promotion
        - Compare outcomes between groups
        - Difference = Average Treatment Effect (ATE)
        
        Advantages:
        - Causality clearly established
        - Unbiased uplift estimates
        - Gold standard for evidence
        
        Disadvantages:
        - Expensive (need large samples)
        - Time-consuming (3-6 months)
        - Business resistance ("Why give control group nothing?")
        - Ethical concerns (withholding potential benefit)
        
        **Observational Approaches** (when RCT not feasible):
        
        1. Propensity Score Matching:
           - Match treated HCPs to similar untreated HCPs
           - Compare outcomes between matched pairs
           - Approximates RCT if matching good
        
        2. Difference-in-Differences:
           - Compare before/after for treated vs control
           - Controls for time trends
           - Assumes parallel trends (often violated)
        
        3. Instrumental Variables:
           - Find variable that affects treatment but not outcome directly
           - Use to isolate causal effect
           - Hard to find valid instruments
        
        4. Machine Learning Uplift Models:
           - Two-model approach: Model treatment and control separately"""
Comprehensive Domain Knowledge and Business Context for Vector Database
Focus: Pharmaceutical commercial analytics domain expertise, industry knowledge,
business reasoning patterns, and interpretive frameworks.

NO SQL - SQL patterns belong in schema_context.py
This file contains ONLY business/domain knowledge for semantic retrieval.
"""

# =============================================================================
# PHARMACEUTICAL COMMERCIAL ANALYTICS DOMAIN KNOWLEDGE
# =============================================================================

DOMAIN_KNOWLEDGE = [
    # Document 1: HCP Targeting Fundamentals
    {
        "id": "domain_hcp_targeting_001",
        "category": "domain_knowledge",
        "subcategory": "hcp_targeting",
        "title": "Healthcare Professional (HCP) Targeting and Segmentation",
        "content": """
        HCP targeting is the foundation of pharmaceutical commercial strategy. Understanding
        how to identify, segment, and prioritize healthcare professionals drives sales effectiveness
        and ROI optimization.
        
        ## Core Concepts
        
        **HCP Definition**: Healthcare professionals who prescribe medications, including:
        - Physicians (MDs, DOs) across all specialties
        - Nurse Practitioners (NPs) with prescribing authority
        - Physician Assistants (PAs) with prescribing authority
        - Specialists vs Generalists (Primary Care)
        
        **Segmentation Approaches**:
        
        1. **Value-Based Tiering**:
           - HIGH/TIER_1: Top 20% of prescribers (often drive 60-80% of volume)
           - MEDIUM/TIER_2: Middle 30% (consistent moderate prescribers)
           - LOW/TIER_3: Bottom 50% (low volume, occasional prescribers)
           
        2. **Decile Analysis**:
           - Rank all HCPs by prescription volume into 10 equal groups
           - Top decile (10%) typically generates 30-40% of total prescriptions
           - Used for resource allocation and targeting strategies
        
        3. **Behavioral Segmentation**:
           - Early Adopters: Quick to try new therapies (5-10% of market)
           - Majority: Wait for evidence accumulation (70-80%)
           - Laggards: Very slow to change prescribing (10-15%)
           - Non-prescribers: Don't prescribe in therapeutic area
        
        4. **Specialty-Based Segmentation**:
           - Specialists: Higher volume in specific therapeutic areas
           - Primary Care: Broader patient base, lower per-patient complexity
           - Academic: Teaching hospitals, research-oriented
           - Community: Private practice, patient satisfaction focus
        
        ## Key Metrics
        
        **Prescription Metrics**:
        - **TRx (Total Prescriptions)**: New prescriptions + refills
        - **NRx (New Prescriptions)**: First-time prescriptions only (leading indicator)
        - **NBRx (New-to-Brand)**: Patients switching from competitor
        - **Refills**: Continuation prescriptions (indicates satisfaction/adherence)
        
        **Market Share Metrics**:
        - **Prescriber Share**: Percentage of HCPs who have ever prescribed your brand
        - **Share of Voice (SOV)**: Your prescriptions / total category prescriptions
        - **Fair Share**: Expected share based on market factors
        - **Excess Share**: Actual share - fair share (indicates targeting effectiveness)
        
        **Targeting Metrics**:
        - **Reach**: Percentage of target HCPs contacted
        - **Frequency**: Average number of contacts per HCP
        - **Coverage**: Percentage of total prescription volume covered by targets
        
        ## Business Context: The 80/20 Rule
        
        Pharmaceutical sales follow extreme Pareto distributions:
        - 20% of HCPs generate 80% of prescriptions
        - Top 10% of HCPs can generate 50-60% of volume
        - Top 1% (super-prescribers) can drive 20-30% of market
        
        **Implications**:
        - Sales resources must concentrate on high-value HCPs
        - Missing one TIER_1 HCP = losing 50-100 TIER_3 HCPs worth of volume
        - Territory alignment critical: ensure no TIER_1 HCP uncovered
        
        ## Prescribing Behavior Patterns
        
        **Specialty Differences**:
        - Cardiologists: Evidence-driven, focus on outcomes (mortality, CV events)
        - Endocrinologists: Metrics-focused (A1C, weight, glucose control)
        - Primary Care: Safety-first, patient convenience, formulary considerations
        - Oncologists: Guidelines-driven, multidisciplinary team decisions
        
        **Experience Level**:
        - Young physicians (0-5 years): More open to new therapies, digitally engaged
        - Mid-career (6-15 years): Established patterns, selective adoption
        - Senior (15+ years): Very set patterns, relationship-driven, skeptical of change
        
        **Practice Setting**:
        - Academic medical centers: Research participation, formulary restrictions, resident influence
        - Large group practices: Standardized protocols, group purchasing, EHR integration
        - Solo/small practices: More autonomy, personal relationships matter, cost-conscious
        
        ## Targeting Challenges
        
        1. **Access Restrictions**: No-see policies, office staff gatekeepers, administrative burden
        2. **Digital Fatigue**: Email overload, declining response rates
        3. **Competitive Noise**: Multiple reps from different companies competing for attention
        4. **Regulatory Constraints**: Anti-kickback laws, sunshine act reporting, promotional limits
        5. **Changing Demographics**: Younger HCPs prefer digital engagement over in-person
        
        ## Success Factors
        
        **High-Performing Targeting Strategies**:
        - Multichannel coordination: Email + call + sample + event
        - Personalization: Message tailored to specialty, practice type, patient demographics
        - Optimal frequency: 2-3 meaningful touchpoints per month (not spam)
        - Value delivery: Clinical data, peer comparisons, practice support
        - Relationship building: Consistency in rep assignment, long-term engagement
        
        **Poor Strategies** (to avoid):
        - Spray and pray: Equal treatment of all HCPs regardless of potential
        - High frequency, low value: Excessive calls with no new information
        - Generic messaging: Same message to cardiologist and primary care
        - Transactional approach: Only contact when launching new product
        """,
        "tags": ["HCP", "targeting", "segmentation", "prescribing_behavior", "TRx", "NRx", "market_share"],
        "use_cases": ["HCP_engagement", "NRx_forecasting", "messaging_optimization"]
    },
    
    # Document 2: Prescription Forecasting
    {
        "id": "domain_prescription_forecasting_001",
        "category": "domain_knowledge",
        "subcategory": "forecasting",
        "title": "Prescription Volume Forecasting in Pharmaceutical Commercial",
        "content": """
        Prescription forecasting predicts future prescription volumes to support commercial planning,
        resource allocation, and financial projections. Accuracy is critical for multi-million
        dollar decisions in sales force sizing, inventory, and marketing spend.
        
        ## Business Context
        
        **Why Forecasting Matters**:
        - **Financial Planning**: Revenue projections for Wall Street guidance
        - **Sales Force Sizing**: How many reps needed to achieve targets?
        - **Territory Design**: Which geographies need coverage?
        - **Marketing Budget**: How much to spend on campaigns?
        - **Supply Chain**: Manufacturing and inventory planning
        - **Quota Setting**: Realistic but challenging targets for reps
        
        **Forecast Horizons**:
        - Short-term (1-3 months): Tactical execution, rep targeting
        - Medium-term (4-12 months): Budget planning, campaign design
        - Long-term (1-5 years): Strategic planning, launches, lifecycle management
        
        ## Forecasting Components
        
        **Time Series Patterns**:
        
        1. **Trend**: Long-term direction (growth, decline, plateau)
           - Mature brands: Flat or declining trend (generic erosion)
           - Growth brands: Positive trend (market expansion, share gains)
           - New launches: Steep S-curve (slow start, rapid growth, plateau)
        
        2. **Seasonality**: Recurring patterns within year
           - Respiratory drugs: Peak flu season (Nov-Mar)
           - Allergy medications: Spring peak (Apr-Jun)
           - Dermatology: Summer peak (skin conditions visible)
           - End-of-year: Deductible resets, holiday lull
        
        3. **Cyclical**: Multi-year patterns
           - Economic cycles: Recession reduces elective prescriptions
           - Insurance cycles: Formulary changes every 2-3 years
           - Competitive cycles: Launches, genericization waves
        
        4. **Random**: Unpredictable variation
           - Unusually severe flu season
           - Sudden safety warnings
           - Celebrity endorsements
           - Viral social media events
        
        ## Key Features for Forecasting
        
        **Historical Prescription Data**:
        - Lagged TRx/NRx (3, 6, 12 months prior)
        - Year-over-year growth rates
        - Moving averages (smooth out noise)
        - Exponentially weighted averages (recent data weighted higher)
        
        **HCP Characteristics**:
        - Specialty (cardiologists prescribe more CV drugs)
        - Patient volume (high-volume HCPs = more prescriptions)
        - Years in practice (established vs new physicians)
        - Geography (urban vs rural, regional variations)
        - Academic affiliation (teaching vs community)
        
        **Promotional Activity**:
        - Sales call frequency (more calls = more prescriptions, up to saturation)
        - Email engagement (open rates, click rates)
        - Sample distribution (free trials drive adoption)
        - Event attendance (speaker programs, conferences)
        - Digital advertising impressions
        
        **Market Dynamics**:
        - Competitor market share (competitive pressure)
        - New competitor launches (erosion risk)
        - Generic entry (catastrophic decline for brands)
        - Formulary status (preferred, non-preferred, prior auth)
        - Pricing changes (copay increases reduce demand)
        
        **External Factors**:
        - Clinical guidelines (new recommendations drive shifts)
        - Safety warnings (black box warnings crash prescriptions)
        - Published studies (positive RCTs boost prescribing)
        - Reimbursement policy (Medicare/Medicaid changes)
        - Disease prevalence (obesity epidemic increases diabetes drugs)
        
        ## Forecasting Challenges
        
        **Data Quality Issues**:
        - Missing data (incomplete call tracking)
        - Delayed reporting (2-4 week lag in prescription data)
        - Data errors (system glitches, duplicate records)
        - Sample bias (only captures retail, not hospital)
        
        **External Shocks**:
        - Unexpected competitor launches
        - Safety warnings (drug withdrawals)
        - Pandemic effects (COVID-19 disrupted routine prescribing)
        - Regulatory changes (sudden formulary restrictions)
        - Supply disruptions (shortages redirect to alternatives)
        
        **Model Limitations**:
        - Overfitting to historical patterns (market changed)
        - Concept drift (relationship between features and prescriptions changed)
        - Black swan events (unpredictable by definition)
        - Attribution errors (correlation vs causation in promotional impact)
        
        ## Accuracy Expectations
        
        **Typical Performance** (MAPE - Mean Absolute Percentage Error):
        - Mature stable brands: 8-15% error
        - Growing brands: 15-25% error
        - New launches: 30-50% error (high uncertainty)
        - Generic entry: Nearly impossible to forecast precisely
        
        **HCP-Level vs Aggregate**:
        - Individual HCP forecasts: ±30-50% error (noisy)
        - Territory-level forecasts: ±15-25% error (averaging effect)
        - National-level forecasts: ±8-15% error (law of large numbers)
        
        ## Business Translation
        
        **Example**: NRx forecast for 10,000 HCPs
        - Model predicts: 50 ±10 NRx per HCP per month
        - Total forecast: 500,000 NRx/month (±100,000 range)
        - At $50 revenue per Rx: $25M monthly revenue (±$5M)
        - Annual: $300M (±$60M range)
        
        **Impact of Accuracy**:
        - 5% RMSE improvement = $15M better annual forecast precision
        - Prevents over-manufacturing (inventory waste)
        - Prevents under-manufacturing (stockouts, lost sales)
        - Enables optimal sales force sizing (avoid over/under-staffing)
        
        ## Use in Commercial Decisions
        
        **Sales Targeting**: Forecast identifies high-potential HCPs
        - HCP predicted 100 NRx/month → High priority target
        - HCP predicted 5 NRx/month → Low priority (limited resources)
        
        **Campaign Planning**: Forecast quantifies promotional impact
        - If 10% of HCPs targeted with campaign
        - Expected lift: +20 NRx per HCP per month
        - ROI: (20 NRx × $50 × 1000 HCPs) / campaign cost
        
        **Territory Alignment**: Forecast balances workload
        - Target: Each rep covers $5M annual forecast potential
        - Adjust boundaries to equalize opportunity
        - Ensures fair quota setting
        """,
        "tags": ["forecasting", "TRx", "NRx", "time_series", "prediction", "accuracy", "business_planning"],
        "use_cases": ["NRx_forecasting", "model_drift_detection"]
    },
    
    # Document 3: Marketing Response Modeling
    {
        "id": "domain_marketing_response_001",
        "category": "domain_knowledge",
        "subcategory": "marketing",
        "title": "Marketing Response and Campaign Effectiveness in Pharma",
        "content": """
        Marketing response modeling predicts and measures how HCPs respond to promotional
        activities. The goal is maximizing ROI by targeting the right HCPs with the right
        messages through the right channels.
        
        ## The Fundamental Challenge
        
        **Not all HCPs are created equal in response to promotion**:
        
        - **Persuadables** (20-30%): Will change behavior if promoted → TARGET THESE
        - **Sure Things** (10-20%): Already prescribing maximally → Don't waste resources
        - **Lost Causes** (40-50%): Won't change regardless of promotion → Avoid
        - **Do Not Disturbs** (<5%): Negative response to promotion → Definitely avoid
        
        **Traditional mistake**: Treating all HCPs equally (spray and pray)
        **Optimal strategy**: Focus limited resources on persuadables
        
        ## Response Types and Measurement
        
        **Engagement Responses** (Behavioral):
        - Email opened (typically 10-20% of recipients)
        - Email clicked (typically 2-5% of recipients)
        - Call answered/accepted (30-50% of attempts)
        - Event attended (5-15% of invitees)
        - Website visited after promotion
        - Sample accepted and used
        
        **Commercial Responses** (Outcome):
        - Prescription increase (10-20% of engaged HCPs)
        - New patient starts (conversion to brand)
        - Share of voice increase (prescribing more vs competitors)
        - Patient volume increase (treating more patients in category)
        - Persistence improvement (patients stay on therapy longer)
        
        **Time to Response**:
        - Immediate (1-7 days): Email open, call engagement
        - Short-term (1-4 weeks): New patient trials
        - Medium-term (1-3 months): Established prescribing pattern change
        - Long-term (3-12 months): Practice-wide protocol adoption
        
        ## Campaign Types and Effectiveness
        
        **1. Email Campaigns**:
        - **Strengths**: Low cost ($0.05-0.50 per email), high reach, measurable
        - **Weaknesses**: Low engagement (80-90% never open), easy to ignore
        - **Best for**: High-volume awareness, specialist targeting, digital-savvy HCPs
        - **Typical ROI**: 3-5x (for well-targeted campaigns)
        
        **2. Sales Representative Calls**:
        - **Strengths**: Personal relationship, two-way dialogue, can overcome objections
        - **Weaknesses**: Expensive ($150-300 per call), access declining, limited reach
        - **Best for**: High-value targets, complex messages, relationship building
        - **Typical ROI**: 5-10x (when targeted properly)
        
        **3. Events (Speaker Programs, Dinners)**:
        - **Strengths**: High engagement, peer influence, memorable experience
        - **Weaknesses**: Very expensive ($500-2000 per attendee), low attendance rates
        - **Best for**: Key opinion leaders, specialty audiences, launch campaigns
        - **Typical ROI**: 2-4x (highly variable by event quality)
        
        **4. Sampling Programs**:
        - **Strengths**: Trial removes prescribing barriers, patient experience drives loyalty
        - **Weaknesses**: Expensive (drug cost + distribution), potential waste
        - **Best for**: New patient acquisition, switching campaigns, formulary obstacles
        - **Typical ROI**: 4-8x (if samples reach right patients)
        
        **5. Digital Advertising**:
        - **Strengths**: Broad reach, retargeting capability, cost-effective impressions
        - **Weaknesses**: Banner blindness, attribution challenges, passive engagement
        - **Best for**: Awareness building, reinforcement, younger HCP audiences
        - **Typical ROI**: 2-3x (lower but highly scalable)
        
        ## Key Response Drivers
        
        **HCP Factors**:
        - Historical engagement rate (past behavior predicts future)
        - Current prescribing level (non-prescribers hardest to convert)
        - Specialty match (relevance to practice)
        - Patient volume (capacity to prescribe more)
        - Practice setting (academic vs community)
        - Communication preferences (email vs call vs event)
        
        **Message Factors**:
        - Relevance to specialty (CV outcomes for cardiologists)
        - Evidence quality (Level 1 RCT vs observational)
        - Newness (novel information vs repetition)
        - Format (video vs text, interactive vs passive)
        - Sender credibility (peer vs pharma rep)
        
        **Timing Factors**:
        - Contact frequency (optimal 2-3x per month, saturation at 5+)
        - Days since last contact (sweet spot: 14-30 days)
        - Day of week (Tuesday-Thursday best for email)
        - Time of day (10am-2pm best for calls)
        - Season (avoid holidays, summer vacations)
        
        **Competitive Factors**:
        - Competitor promotional intensity (noise level)
        - Recent competitor launches (defensive positioning needed)
        - Market share dynamics (leadership vs challenger strategies)
        
        ## Attribution Challenges
        
        **The Problem**: HCPs receive multiple touchpoints, which one caused response?
        
        **Attribution Models**:
        - **First-touch**: Credit to first interaction (undervalues nurturing)
        - **Last-touch**: Credit to final interaction before prescription (ignores journey)
        - **Linear**: Equal credit to all touchpoints (oversimplifies)
        - **Time-decay**: More credit to recent interactions (reasonable compromise)
        - **Data-driven**: ML model learns optimal attribution (best but complex)
        
        **Multi-Channel Synergies**:
        - Email + Call: 1.5-2x lift vs either alone
        - Call + Sample: 2-3x lift (trial removes barriers)
        - Email + Event: 1.3-1.7x lift (reinforcement effect)
        - Digital + Call: 1.2-1.5x lift (awareness + action)
        
        ## Diminishing Returns and Saturation
        
        **Promotional Frequency Curve**:
        - 0 contacts: Baseline prescribing
        - 1 contact/month: +15% lift
        - 2 contacts/month: +25% lift (optimal for most)
        - 3 contacts/month: +30% lift (marginal gain)
        - 4+ contacts/month: +30-35% lift (saturation, no additional benefit)
        - 6+ contacts/month: Potential negative (annoyance, opt-outs)
        
        **Channel-Specific Saturation**:
        - Email: Can tolerate 2-3 per month
        - Calls: Maximum 3-4 per month
        - Events: 1-2 per quarter
        - Samples: Ongoing (as needed for patients)
        
        ## ROI Calculation Framework
        
        **Campaign ROI Formula**:
        ROI = (Incremental Revenue - Campaign Cost) / Campaign Cost
        
        **Example - Email Campaign**:
        - Universe: 10,000 HCPs
        - Email cost: $5,000 (including design, deployment)
        - Response rate: 15% engage
        - Conversion rate: 20% of engaged increase prescriptions
        - Incremental prescriptions: 10,000 × 0.15 × 0.20 × 5 NRx = 1,500 NRx
        - Revenue: 1,500 × $50 = $75,000
        - ROI: ($75,000 - $5,000) / $5,000 = 14x
        
        **Example - Sales Call Campaign**:
        - Targeted HCPs: 500 (high-value segment)
        - Calls: 3 per HCP over quarter = 1,500 total calls
        - Cost: 1,500 × $200 = $300,000
        - Response rate: 40% increase prescriptions
        - Incremental prescriptions: 500 × 0.40 × 15 NRx = 3,000 NRx
        - Revenue: 3,000 × $50 = $150,000
        - ROI: ($150,000 - $300,000) / $300,000 = -0.5x (NEGATIVE!)
        
        **Key Insight**: High-touch is NOT always better. ROI depends on targeting precision.
        
        ## A/B Testing Best Practices
        
        **Randomization**:
        - Split similar HCPs into treatment vs control
        - Control for confounders (specialty, volume, geography)
        - Ensure sufficient sample size (power analysis)
        
        **What to Test**:
        - Message content (clinical data vs patient stories)
        - Channel (email vs call vs event)
        - Frequency (1x vs 2x vs 3x per month)
        - Timing (day of week, time of day)
        - Format (video vs text, long vs short)
        
        **Measurement Period**:
        - Short campaigns: 1-3 month measurement window
        - Long campaigns: 3-6 month measurement window
        - Account for lag (prescriptions take time to reflect behavior change)
        
        ## Business Impact
        
        **Optimized vs Non-Optimized Campaigns**:
        - Traditional spray-and-pray: 10% response rate, 2x ROI
        - Model-optimized targeting: 25% response rate (targeting persuadables), 6x ROI
        - Net impact: 3x improvement in marketing effectiveness
        - At $10M annual marketing budget: $20M additional revenue from optimization
        """,
        "tags": ["marketing", "campaigns", "response", "ROI", "engagement", "targeting", "channels"],
        "use_cases": ["HCP_engagement", "messaging_optimization", "NRx_forecasting"]
    },
    
    # Document 4: Feature Importance and Driver Analysis
    {
        "id": "domain_feature_importance_001",
        "category": "domain_knowledge",
        "subcategory": "feature_analysis",
        "title": "Understanding Prediction Drivers and Feature Importance",
        "content": """
        Feature importance analysis answers the critical question: "WHY does the model predict
        what it predicts?" Understanding drivers is essential for translating model outputs
        into actionable business strategies.
        
        ## Why Feature Importance Matters
        
        **Business Value**:
        - **Strategy**: Know which levers to pull (focus on controllable high-impact features)
        - **Resource Allocation**: Invest in activities that drive outcomes
        - **Message Development**: Emphasize features that influence prescribing
        - **Targeting Rules**: Create business rules from data patterns
        - **Model Validation**: Ensures model learns sensible relationships
        - **Stakeholder Buy-in**: Explainable models build trust
        
        **Example Impact**:
        If model shows "call_frequency" is rank 2 driver:
        → Actionable: Increase calls to high-potential HCPs
        If model shows "hcp_age" is rank 2 driver:
        → Not actionable: Can't change HCP age, but can segment strategy by age
        
        ##Types of Feature Importance
        
        **1. Gain/Gini Importance** (Tree-based models):
        - Measures cumulative improvement in prediction when splitting on feature
        - Interpretation: "How much does this feature improve model accuracy?"
        - Strengths: Fast to calculate, available in all tree models
        - Weaknesses: Biased toward high-cardinality features (many unique values)
        - Best for: Quick insights, relative ranking of features
        
        **2. Permutation Importance**:
        - Measures performance drop when feature values randomly shuffled
        - Interpretation: "How much does model rely on this feature?"
        - Strengths: Model-agnostic, unbiased
        - Weaknesses: Slow for large datasets, unstable with correlated features
        - Best for: Confirming importance of suspicious features
        
        **3. SHAP (Shapley Values)**:
        - Measures average marginal contribution of feature across all predictions
        - Interpretation: "How much does this feature contribute to each prediction?"
        - Strengths: Theoretically sound, shows direction (positive/negative), handles interactions
        - Weaknesses: Computationally expensive, complex to explain to stakeholders
        - Best for: Detailed analysis, individual prediction explanations
        
        **4. Coefficient Magnitude** (Linear models):
        - Absolute value of regression coefficients
        - Interpretation: "Linear effect size per unit change"
        - Strengths: Simple, interpretable, shows direction
        - Weaknesses: Only captures linear relationships, sensitive to scaling
        - Best for: Simple models, linear relationships
        
        ## Common Pharma Feature Patterns
        
        **Historical Behavior Features** (Usually Top 5):
        - lagged_nrx_3mo: Past 3-month prescription average
        - lagged_nrx_6mo: Past 6-month prescription average
        - nrx_trend: Month-over-month growth rate
        - peak_nrx_ever: Historical maximum (prescribing potential)
        
        **Why Important**: Past behavior is strongest predictor of future behavior
        
        **HCP Characteristic Features**:
        - hcp_specialty: Different specialties have different prescribing patterns
        - patient_volume: High-volume HCPs prescribe more (capacity)
        - years_in_practice: Experience correlates with prescribing patterns
        - geography: Urban vs rural, regional variations
        - academic_affiliation: Teaching vs community practice differences
        
        **Why Important**: Segment-specific strategies needed (one size doesn't fit all)
        
        **Promotional Activity Features**:
        - call_frequency: Number of sales rep visits (controllable!)
        - email_open_rate: Engagement indicator (digital responsiveness)
        - samples_provided: Trial removes barriers to prescribing
        - event_attendance: High-touch engagement signal
        
        **Why Important**: These are ACTIONABLE - we can directly influence them
        
        **Market Context Features**:
        - competitor_market_share: Competitive pressure level
        - formulary_tier: Reimbursement favorability
        - days_since_launch: Product maturity/familiarity
        - clinical_guideline_recommendation: External validation
        
        **Why Important**: External factors beyond our control but critical for realistic expectations
        
        ## Feature Interactions
        
        **Synergistic Interactions** (1 + 1 = 3):
        - specialty × patient_volume: High-volume specialists are MUCH more valuable
        - call_frequency × email_open_rate: Multi-channel synergy
        - samples × formulary_tier: Samples overcome cost barriers
        - academic_affiliation × years_in_practice: Senior academic leaders are influencers
        
        **Antagonistic Interactions** (1 + 1 = 1):
        - high_call_frequency × low_email_opens: Engagement fatigue cancels out
        - specialty_mismatch × message_content: Wrong message reduces all impact
        - high_competitor_share × low_promotion: Can't overcome entrenched competition cheaply
        
        **Threshold Effects**:
        - Minimum call frequency (2/month) needed for impact, below that = zero effect
        - Formulary tier: No amount of promotion overcomes non-covered status
        - Patient volume floor: Very low-volume HCPs can't prescribe much regardless
        
        ## Interpreting Feature Importance Rankings
        
        **Example Rankings and Meaning**:
        
        Rank 1: lagged_nrx_6mo (Importance: 0.30)
        → Meaning: Past prescribing drives 30% of prediction power
        → Business insight: Focus on HCPs with established prescribing patterns
        → Action: Nurture existing prescribers, prevent erosion
        
        Rank 2: hcp_specialty (Importance: 0.18)
        → Meaning: Specialty explains 18% of variance
        → Business insight: Different specialties need different strategies
        → Action: Develop specialty-specific messages and targets
        
        Rank 3: call_frequency (Importance: 0.15)
        → Meaning: Promotional calls drive 15% of prediction
        → Business insight: Calls matter! But not overwhelming (only 15%)
        → Action: Optimize call allocation, don't overdo it
        
        Rank 4: patient_volume (Importance: 0.12)
        → Meaning: High-volume HCPs can prescribe more
        → Business insight: Capacity constraints matter
        → Action: Prioritize high-volume HCPs in targeting
        
        Rank 5: competitor_market_share (Importance: 0.10)
        → Meaning: Competitive dynamics influence prescribing
        → Business insight: Harder to grow in competitive markets
        → Action: Increase share-of-voice in competitive territories
        
        Ranks 6-20: Various features (Importance: 0.15 combined)
        → Meaning: Many small factors contribute
        → Business insight: No silver bullet, multiple levers matter
        
        ## Translating to Business Action
        
        **Framework**:
        1. Identify top 10 features
        2. Classify: Controllable vs Non-controllable
        3. For controllable: What action increases/optimizes this feature?
        4. For non-controllable: How do we segment/adapt strategy?
        5. Estimate ROI of actions
        6. Prioritize high-impact, low-cost actions
        
        **Example Translation**:
        
        Feature: call_frequency (Rank 3, controllable)
        - Current state: Average 2 calls/HCP/month across all HCPs
        - Opportunity: Model shows optimal frequency is 3-4 calls for high-potential HCPs
        - Action: Increase calls to top 20% HCPs from 2 to 4 per month
        - Cost: 2,000 HCPs × 2 extra calls × $200 = $800K
        - Benefit: 2,000 HCPs × 10 incremental NRx × $50 = $1M
        - ROI: ($1M - $800K) / $800K = 25%
        - Decision: Implement
        
        Feature: hcp_specialty (Rank 2, non-controllable)
        - Current state: Same message to all specialties
        - Opportunity: Cardiologists respond to CV outcomes, PCPs to safety
        - Action: Develop 3 specialty-specific message variants
        - Cost: $50K message development + $20K targeting system
        - Benefit: 15% improvement in engagement rate × baseline impact
        - ROI: Estimated