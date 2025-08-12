import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import shap
from collections import defaultdict

# ---- Simulate example data ----
np.random.seed(42)
n_agents = 100
n_behaviors = 30

# Numeric behavioral features (counts * confidence)
behavior_cols = [f"behavior_{i}" for i in range(1, n_behaviors+1)]
X_behaviors = pd.DataFrame(np.random.rand(n_agents, n_behaviors), columns=behavior_cols)

# Categorical features: tenure (buckets), ranking (buckets)
tenure_buckets = ['0-6m', '6-12m', '1-2y', '2y+']
ranking_buckets = ['low', 'mid', 'high']

X_cats = pd.DataFrame({
    'tenure': np.random.choice(tenure_buckets, size=n_agents),
    'ranking': np.random.choice(ranking_buckets, size=n_agents)
})

# Combine
X_raw = pd.concat([X_behaviors, X_cats], axis=1)

# Binary target (simulate some signal)
y = (X_behaviors.behavior_1 * 2 + X_behaviors.behavior_5 * -1 + 
     (X_cats.tenure == '2y+').astype(int)*0.5 + np.random.normal(0, 0.5, n_agents)) > 1
y = y.astype(int)

# ---- Define preprocessing pipeline ----
categorical_cols = ['tenure', 'ranking']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), behavior_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# ---- Pipelines for models ----
logreg_pipe = Pipeline([
    ('preproc', preprocessor),
    ('logreg', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000))
])

tree_pipe = Pipeline([
    ('preproc', preprocessor),
    ('tree', DecisionTreeRegressor(random_state=42))
])

# ---- Fit logistic regression on full data ----
logreg_pipe.fit(X_raw, y)
logreg_coefs = logreg_pipe.named_steps['logreg'].coef_[0]
feature_names = logreg_pipe.named_steps['preproc'].get_feature_names_out()

# Map feature name to coefficient for quick lookup
logreg_coef_dict = dict(zip(feature_names, logreg_coefs))

# ---- Function to train tree and get SHAP values ----
def train_tree_and_shap(X_train, y_train, X_eval):
    tree_pipe.fit(X_train, y_train)
    tree_model = tree_pipe.named_steps['tree']
    X_eval_transformed = tree_pipe.named_steps['preproc'].transform(X_eval)
    explainer = shap.TreeExplainer(tree_model)
    shap_vals = explainer.shap_values(X_eval_transformed)
    return shap_vals, X_eval_transformed

# ---- Bootstrap Stability Check ----
n_bootstraps = 30
top_k = 5
agent_indices = X_raw.index.tolist()

# Dict: agent_id -> feature -> count of times in top_k with sign agreement
stability_counts = {agent: defaultdict(int) for agent in agent_indices}

print("Starting bootstrap stability check...")
for b in range(n_bootstraps):
    X_boot, y_boot = resample(X_raw, y, replace=True, stratify=y, random_state=42 + b)
    shap_vals, X_eval_transformed = train_tree_and_shap(X_boot, y_boot, X_raw)
    
    # shap_vals shape: (n_samples, n_features)
    # For regression, shap_values is 2D array: (samples, features)
    for i, agent_id in enumerate(agent_indices):
        shap_for_agent = shap_vals[i]
        
        # Build DataFrame with features and shap values
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_for_agent
        })
        
        # Filter out tenure and ranking features from recommendations
        tenure_features = [f for f in feature_names if f.startswith('cat__tenure')]
        ranking_features = [f for f in feature_names if f.startswith('cat__ranking')]
        exclude_feats = set(tenure_features + ranking_features)
        df_shap = df_shap[~df_shap['feature'].isin(exclude_feats)]
        
        # Check sign agreement with logistic regression coefficients
        df_shap['logreg_coef'] = df_shap['feature'].map(logreg_coef_dict)
        df_shap['sign_agree'] = np.sign(df_shap['shap_value']) == np.sign(df_shap['logreg_coef'])
        
        # Keep only features where signs agree and coef != 0
        df_shap_filtered = df_shap[(df_shap['sign_agree']) & (df_shap['logreg_coef'] != 0)]
        
        # Take top_k by absolute shap value
        top_feats = df_shap_filtered.reindex(df_shap_filtered['shap_value'].abs().sort_values(ascending=False).index).head(top_k)
        
        for feat in top_feats['feature']:
            stability_counts[agent_id][feat] += 1
    
    print(f"Bootstrap {b+1}/{n_bootstraps} done.")

# ---- Final stable recommendations threshold (70%) ----
stable_recommendations = {}
for agent_id, feat_dict in stability_counts.items():
    stable_feats = [feat for feat, count in feat_dict.items() if count / n_bootstraps >= 0.7]
    stable_recommendations[agent_id] = stable_feats

# ---- Generate final recommendations per agent ----
final_recs = []
X_preprocessed = tree_pipe.named_steps['preproc'].transform(X_raw)
shap_vals_final = shap.TreeExplainer(tree_pipe.named_steps['tree']).shap_values(X_preprocessed)

for i, agent_id in enumerate(agent_indices):
    shap_for_agent = shap_vals_final[i]
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_for_agent
    })
    tenure_features = [f for f in feature_names if f.startswith('cat__tenure')]
    ranking_features = [f for f in feature_names if f.startswith('cat__ranking')]
    exclude_feats = set(tenure_features + ranking_features)
    df_shap = df_shap[~df_shap['feature'].isin(exclude_feats)]
    df_shap['logreg_coef'] = df_shap['feature'].map(logreg_coef_dict)
    
    # Only keep stable features
    stable_feats = stable_recommendations.get(agent_id, [])
    df_stable = df_shap[df_shap['feature'].isin(stable_feats)].copy()
    
    # Separate positive and negative contributions
    positive_feats = df_stable[df_stable['shap_value'] > 0].sort_values('shap_value', ascending=False)['feature'].tolist()
    negative_feats = df_stable[df_stable['shap_value'] < 0].sort_values('shap_value')['feature'].tolist()
    
    # Predict probability by logistic regression for reference
    prob = logreg_pipe.predict_proba(X_raw.loc[[agent_id]])[0,1]
    
    final_recs.append({
        'agent_id': agent_id,
        'probability': prob,
        'keep_doing': positive_feats[:5],  # top 5 positive drivers
        'improve': negative_feats[:5]      # top 5 negative drivers
    })

final_recs_df = pd.DataFrame(final_recs)
print("\nSample final recommendations:")
print(final_recs_df.head())
