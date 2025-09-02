
"""
Fallstudie Model Engineering
    Aufgabenstellung 1: Erstellen eines Prognosemodells 
    des Kreditkartenzahlungsverkehr für Online-Einkäufe

@author: Quang Vu Pham
Matrikelnr. 9221131
"""

# IMPORT BIBLIOTHEKS & DATA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, auc
)

df = pd.read_csv(
    r'C:\Backup_VP\FYI\Studium\Fallstudie Model Engineering\data.csv', delimiter=";", decimal = ",", 
    dtype={
        "tmsp": "float",
        "country": "string",
        "amount": "float",         
        "success": "bool",
        "PSP": "string",
        "3D_secured": "bool",
        "card": "string"
    },
)

#  DATA PREPARATION
# Data cleaning
df = df.drop(df.columns[0], axis=1)

# Convert data type
df['tmsp'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['tmsp'], unit='D')
df['3D_secured'] = df['3D_secured'].astype(int)
df['success'] = df['success'].astype(int)

# Add new features
df["transaction_hour"] = df["tmsp"].dt.hour
df["transaction_weekday"] = df["tmsp"].dt.dayofweek  # Monday=0


# Add payment_id according to the logic: same country, same amount, same card, within 1 min
df.loc[0, 'payment_id'] = 1
j=1

for i in range(len(df.index)-1):
    if df.iat[i+1,1] == df.iat[i,1] and df.iat[i+1,2] == df.iat[i,2] and df.iat[i+1,6] == df.iat[i,6] and df.iat[i+1, 0] - df.iat[i, 0] <= pd.Timedelta(minutes=1):
        df.loc[i+1, 'payment_id'] = j
    else:
        j += 1
        df.loc[i+1, 'payment_id'] = j

# Outlier analysis
# Identify payment_ids where the sum of success > 1 and remove them
payment_success_sum = df.groupby('payment_id')['success'].sum()
payment_ids_to_remove = payment_success_sum[payment_success_sum > 1].index
df = df[~df['payment_id'].isin(payment_ids_to_remove)]

# VISUALISATION
# Categorical features
success_rate_country = df.groupby("country")["success"].mean().reset_index()
success_rate_psp = df.groupby("PSP")["success"].mean().reset_index()
success_rate_3D_secured = df.groupby("3D_secured")["success"].mean().reset_index()
success_rate_card = df.groupby("card")["success"].mean().reset_index()

def plot_double_and_rate(df, feature):

    fig, axes = plt.subplots(1, 2, figsize=(12,6), tight_layout=True)

    # Left plot: counts split by success
    ctab = pd.crosstab(df[feature], df["success"])
    ctab.plot(
        kind="bar",
        ax=axes[0],
        edgecolor="black",
        legend=False
    )
    axes[0].set_title(f"Counts by {feature}", fontsize=13)
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Right plot: success rate
    rate = df.groupby(feature)["success"].mean().reset_index()
    sns.barplot(x=feature, y="success", data=rate, edgecolor="black", ax=axes[1])
    axes[1].set_title(f"Success rate by {feature}", fontsize=13)
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Success rate")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].bar_label(axes[1].containers[0], fontsize=10);
    axes[1].set_ylim(0,1)

    axes[0].legend(title="success", labels=["False", "True"])

    plt.show()


plot_double_and_rate(df, "PSP")
plot_double_and_rate(df, "country")
plot_double_and_rate(df, "3D_secured")
plot_double_and_rate(df, "card")


# Numerical features
num_features = ["amount"]
summary_df = df[num_features].describe().drop("count", axis=0)

# Timestamp feature
# Compute success rate
hourly_success = df.groupby("transaction_hour")["success"].mean()
weekday_success = df.groupby("transaction_weekday")["success"].mean()

#Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Lineplot by hour
axes[0].plot(hourly_success.index, hourly_success.values, marker="o")
axes[0].set_title("Success Rate by Transaction Hour")
axes[0].set_xlabel("transaction_hour")
axes[0].set_ylabel("Success Rate")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Lineplot by weekday
weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[1].plot(weekday_success.index, weekday_success.values, marker="o", color="orange")
axes[1].set_title("Success Rate by Weekday")
axes[1].set_xlabel("transaction_weekday")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(weekday_labels)
axes[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# FEATURE ENGINEERING
# Split training and test data according to payment_id for evaluation purposes
unique_payment_ids = df['payment_id'].unique()
train_payment_ids, test_payment_ids = train_test_split(unique_payment_ids, test_size=0.2, random_state=42)
train_df = df[df['payment_id'].isin(train_payment_ids)]
test_df = df[df['payment_id'].isin(test_payment_ids)]

# Feature selection
feature_cols = ['country', 'PSP', 'card', 'amount', '3D_secured','transaction_hour','transaction_weekday'] 
X_train = train_df[feature_cols]
y_train = train_df['success']
X_test = test_df[feature_cols]
y_test = test_df['success']

# Encoding
categorical_features = ['country', 'PSP', 'card']
numerical_features = ['amount', '3D_secured','transaction_hour','transaction_weekday']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)
preprocessor.fit(X_train)
columns = preprocessor.get_feature_names_out()
columns = list(map(lambda x: str(x).split("__")[-1], columns))

X_train = pd.DataFrame(preprocessor.transform(X_train), columns=columns)
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=columns)



# MODEL
# Models with default hyperparameters
classifiers = {
    "dtree": DecisionTreeClassifier(),
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier()
}

def evaluate_model(y_true, y_pred):
    all_metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    return all_metrics

model_performances = []

for label, model in tqdm(classifiers.items()):
    n_splits = 3 # 3 folds
    kf = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42)
    performances = np.zeros((n_splits, 4))  

    X_values = X_train.values
    y_values = y_train.values
    i = 0
    for train_idx, test_idx in kf.split(X_values, y_values):
        train_set = (X_values[train_idx], y_values[train_idx])  
        test_set = (X_values[test_idx], y_values[test_idx])     
        model.fit(*train_set)
        y_pred = model.predict(test_set[0])
        perf = evaluate_model(test_set[1], y_pred)
        performances[i, :] = list(perf.values())
        i += 1
        
    model_performances.append(
        pd.Series(np.mean(performances, axis=0), index=list(perf.keys()), name=label)
    )
    
# Concatenate the results
performances_df = pd.concat(model_performances, axis=1)

# Visualisation
avg_f1 = performances_df.loc["f1"].mean()
fig, ax = plt.subplots()
performances_df.T.plot(
    kind="bar",
    title="Performance of models",
    colormap=plt.cm.viridis,
    width=0.8,
    figsize=(10, 4),
    ax=ax,
)
ylim = ax.get_ylim()
ax.set(ylim=(0, ylim[-1] + 0.06))
ax.hlines(avg_f1, *ax.get_xlim(), ls="--", label="avg_f1", lw=1.2)
ax.legend(
    loc="best",
    shadow=True,
    frameon=True,
    facecolor="inherit",
    bbox_to_anchor=(0.15, 0.01, 1, 1),
    title="Metrics",
)
plt.show()


# GridSearchCV
# Define the hyperparameter grids for Random Forest and XGBoost
param_grid_rf = {
    'n_estimators': [200, 300, 400],
    'max_depth': [5, 7, 9],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [3, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False],
    'class_weight': ['balanced']
}
param_grid_xgb = {
    'n_estimators': [400, 600],
    'max_depth': [4, 5],
    'learning_rate': [0.01, 0.05,  0.1],
    'subsample': [0.6, 0.7],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [2, 3],
    'reg_alpha': [1],
    'reg_lambda': [1],
    'scale_pos_weight': [3, 4]
}

# Perform Grid Search for Random Forest
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)

# Perform Grid Search for XGBoost
grid_search_xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid_xgb,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)
grid_search_xgb.fit(X_train, y_train)


print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best score for Random Forest:", grid_search_rf.best_score_)

"""
Best parameters for Random Forest: {'bootstrap': False,
                                    'class_weight': 'balanced', 
                                    'max_depth': 7, 
                                    'max_features': 'sqrt', 
                                    'min_samples_leaf': 4, 
                                    'min_samples_split': 5, 
                                    'n_estimators': 300}
Best score for Random Forest: 0.4066396117394448
"""

print("Best parameters for XGBoost:", grid_search_xgb.best_params_)
print("Best score for XGBoost:", grid_search_xgb.best_score_)

"""
Best parameters for XGBoost: {'colsample_bytree': 0.8, 
                              'learning_rate': 0.01, 
                              'max_depth': 5, 
                              'min_child_weight': 3, 
                              'n_estimators': 400, 
                              'reg_alpha': 1, 
                              'reg_lambda': 1, 
                              'scale_pos_weight': 4, 
                              'subsample': 0.6}
Best score for XGBoost: 0.40752498684440197
"""

best_models = {
    "rf": grid_search_rf.best_estimator_,
    "xgb": grid_search_xgb.best_estimator_
}

# Cross validation report
for label, model in tqdm(best_models.items()):
    n_splits = 3
    kf = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42)
    performances = np.zeros((n_splits, 4))  # Array to store metrics for each fold

    X_values = X_train.values
    y_values = y_train.values
    i = 0
    for train_idx, test_idx in kf.split(X_values, y_values):
        train_set = (X_values[train_idx], y_values[train_idx])  
        test_set = (X_values[test_idx], y_values[test_idx])     
        model.fit(*train_set)
        y_pred = model.predict(test_set[0])
        perf = evaluate_model(test_set[1], y_pred)
        performances[i, :] = list(perf.values())
        i += 1
        
    model_performances.append(
        pd.Series(np.mean(performances, axis=0), index=list(perf.keys()), name=label)
    )

model_performances_df = pd.DataFrame(model_performances)



# EVALUATION
# Evaluation transaction-wise
best_models = {
    "dtree": DecisionTreeClassifier(),
    "rf": grid_search_rf.best_estimator_,
    "xgb": grid_search_xgb.best_estimator_
}


test_performances = []
confusion_matrices = {}
roc_curves = {}

for label, model in best_models.items():
    model.fit(X_train, y_train)  
    y_pred_test = model.predict(X_test) 
    perf_test = evaluate_model(y_test, y_pred_test)  
    test_performances.append(
        pd.Series(perf_test, name=label)
    )
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    confusion_matrices[label] = cm
    
    # ROC curve
    y_prob_test = model.predict_proba(X_test)[:, 1]  # Predict probabilities
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    roc_curves[label] = (fpr, tpr, roc_auc)   

test_performances_df = pd.concat(test_performances, axis=1)

# Plot confusion matrices and ROC curves for each model
for label in best_models.keys():
    cm = confusion_matrices[label]
    fpr, tpr, roc_auc = roc_curves[label]
    
    plt.figure(figsize=(16, 6))
    
    # Plot confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Success', 'Success'],
                yticklabels=['Not Success', 'Success'])
    plt.title(f'Confusion Matrix for {label}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'ROC Curve for {label}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    
    plt.show()

# Feature Importance
feature_importance_dict = {}

for name, model in best_models.items():
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances = importances.sort_values(ascending=False)
        feature_importance_dict[name] = importances

feature_importance_df = pd.DataFrame(feature_importance_dict)

        
# Ranking der PSPs
# Change test_df from transaction-wise to payment-wise 
# Create binary flag
test_df['psp_success'] = np.where(test_df['success'] == 1, 1, 0)
test_df['psp_fail'] = np.where(test_df['success'] != 1, 1, 0)

# Aggregate successes and failures of each PSP
success_test_df = pd.pivot_table(test_df, index='payment_id', 
                            columns='PSP', 
                            values='psp_success', 
                            aggfunc='sum', 
                            fill_value=0)

fail_test_df = pd.pivot_table(test_df, index='payment_id', 
                         columns='PSP', 
                         values='psp_fail', 
                         aggfunc='sum', 
                         fill_value=0)

# Rename columns 
success_test_df.columns = [f'{psp}_success' for psp in success_test_df.columns]
fail_test_df.columns = [f'{psp}_fail' for psp in fail_test_df.columns]

# Remove unnecessary columns
payment_test_df = test_df.drop(columns=['tmsp', 'success', 'PSP', 'psp_success','psp_fail'])

# Transform: 'first' & 'max' because the values from each trans within a payment are all the same 
payment_test_df = test_df.groupby('payment_id').agg({
    'amount': 'first',
    'country': 'first',
    'card': 'first',
    '3D_secured': 'first',
    'transaction_hour': 'max',
    'transaction_weekday': 'max',
}).reset_index()

# Merge in the new PSP indicators
payment_test_df = payment_test_df.merge(success_test_df, on='payment_id', how='left')
payment_test_df = payment_test_df.merge(fail_test_df, on='payment_id', how='left')

# Flag 'Succeeded', 'Unknown', 'Failed'
payment_test_df['Goldcard_test'] = np.select(
    [payment_test_df['Goldcard_success'] > 0,(payment_test_df['Goldcard_success'] + payment_test_df['Goldcard_fail']) == 0],
    ['Succeeded','Unknown'], default='Failed')

payment_test_df['Moneycard_test'] = np.select(
    [payment_test_df['Moneycard_success'] > 0,(payment_test_df['Moneycard_success'] + payment_test_df['Moneycard_fail']) == 0],
    ['Succeeded','Unknown'], default='Failed')

payment_test_df['UK_Card_test'] = np.select(
    [payment_test_df['UK_Card_success'] > 0,(payment_test_df['UK_Card_success'] + payment_test_df['UK_Card_fail']) == 0],
    ['Succeeded','Unknown'], default='Failed')

payment_test_df['Simplecard_test'] = np.select(
    [payment_test_df['Simplecard_success'] > 0,(payment_test_df['Simplecard_success'] + payment_test_df['Simplecard_fail']) == 0],
    ['Succeeded','Unknown'], default='Failed')

# Drop unnecessary columns
payment_test_df = payment_test_df.drop(columns=['Goldcard_success',
'Moneycard_success', 'Simplecard_success', 'UK_Card_success',
'Goldcard_fail', 'Moneycard_fail', 'Simplecard_fail', 'UK_Card_fail'])

# Test data preparation
# X_payment_test is payment-wise
# All features without PSP
feature_cols_eva = ['country', 'card', 'amount', '3D_secured','transaction_hour','transaction_weekday'] 
X_payment_test = payment_test_df[feature_cols_eva]

psps = ['Goldcard', 'Moneycard', 'UK_Card', 'Simplecard']

results = []

# Iterate through each row in the test set
for index, context_row in X_payment_test.iterrows():
    # Iterate through each PSP
    for psp in psps:
        # Create a copy of the context_row
        row = context_row.copy()
        row['PSP'] = psp  # Set the PSP feature
        
        # Create a DataFrame for prediction
        x_df = pd.DataFrame([row])[feature_cols]
        x_df = pd.DataFrame(preprocessor.transform(x_df), columns=columns)
        # Iterate through each model
        for model_name, model in best_models.items():
            probability = model.predict_proba(x_df)[0, 1]  # Get probability for class 1 (success == 1)

            # Append the result to the results list
            results.append({
                'Index': index,  # Keep track of the original row index for merging to payment_test_df
                'PSP': psp,
                'Model': model_name,
                'Probability of Success': probability
            })


results_df = pd.DataFrame(results)
results_df_pivot = results_df.pivot_table(values='Probability of Success', index='Index', columns=['PSP', 'Model'], aggfunc='first')
# Rename columns to the desired format
results_df_pivot.columns = [f"p_{model}_{psp}" for psp, model in results_df_pivot.columns]

# Merge results_df_pivot to payment_test_df on key = Index
merged_df = payment_test_df.merge(results_df_pivot, left_index=True, right_index=True)
models = ['dtree', 'rf', 'xgb']

# Iterate through each model to replace probabilities with ranks
for model in models:
    # Extract the probability columns for the current model
    prob_columns = [f'p_{model}_{psp}' for psp in psps]
    
    if model == 'dtree':
        # Logic for the dtree model: if 0, all Rank 4
        merged_df[prob_columns] = merged_df[prob_columns].apply(
            lambda row: row.rank(method='min', ascending=False).astype(int).where(row > 0, 4),
            axis=1
        )
    else:
        # Logic for other models: Rank 1 = highest p
        merged_df[prob_columns] = merged_df[prob_columns].apply(
            lambda row: row.rank(method='min', ascending=False).astype(int),
            axis=1
        )
        
# Evaluation payment-wise
# Evaluation matrix
evaluation_scores = {
    'Failed': {1: -1, 2: -0.5, 3: -0.25, 4: 0},
    'Unknown': {1: 2.5, 2: 1.5, 3: 1, 4: 0.5},
    'Succeeded': {1: 4, 2: 3, 3: 2, 4: 1}
}

merged_df_eva =  merged_df

# Replace ranks with corresponding scores based on the test outcome
for model in models:
    for psp in psps:
        rank_column = f'p_{model}_{psp}'
        
        merged_df_eva[rank_column] = merged_df_eva.apply(
            lambda row: evaluation_scores[row[f'{psp}_test']][row[rank_column]] if row[rank_column] in evaluation_scores[row[f'{psp}_test']] else row[rank_column],
            axis=1
        )

# Calculate sum_metric
score_dtree = merged_df_eva[[f'p_dtree_{psp}' for psp in psps]].sum(axis=1)
score_rf = merged_df_eva[[f'p_rf_{psp}' for psp in psps]].sum(axis=1)
score_xgb = merged_df_eva[[f'p_xgb_{psp}' for psp in psps]].sum(axis=1)

total_score_dtree = score_dtree.sum()
total_score_rf = score_rf.sum()
total_score_xgb = score_xgb.sum()
