
from dataloader import DataLoader

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import matplotlib.pyplot as plt

print("Loading data...")
data = DataLoader(training_dir="Training", target_per_class=100000)


# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=10000,
    learning_rate=0.5,
    max_depth=10,
    min_child_weight=1,  
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    scale_pos_weight=1,
    random_state=42,
    eval_metric="error",

)

print("Training model...")

# Train the model and record eval history
xgb_model.fit(
    data.X_train, data.y_train,
    eval_set=[(data.X_train, data.y_train), (data.X_test, data.y_test)]
)

print("Evaluating model...")
# Predict on validation set
y_pred_test = xgb_model.predict(data.X_test)
y_pred_proba_test = xgb_model.predict_proba(data.X_test)[:, 1]
# Evaluate the model
val_accuracy = accuracy_score(data.y_test, y_pred_test)
val_precision = precision_score(data.y_test, y_pred_test)
val_recall = recall_score(data.y_test, y_pred_test)

# Print validation metrics
print(f"Validation Metrics:")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
xgb.plot_importance(xgb_model, max_num_features=20)
plt.title("Top 20 Feature Importance in XGBoost Model")
plt.tight_layout()
plt.show()

xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=20)
plt.title("Top 20 Feature Importance in XGBoost Model (by weight)")

# Calculate and print average feature importance
importance_dict = xgb_model.get_booster().get_score(importance_type='weight')
if importance_dict:
    avg_importance = sum(importance_dict.values()) / len(importance_dict)
else:
    avg_importance = 0.0
print(f"Average feature importance: {avg_importance:.4f}")

# Save the trained model to a file for later reloading
model_dir = "Models"
os.makedirs(model_dir, exist_ok=True)
model_filename = os.path.join(model_dir, "xgb_model.model")
xgb_model.save_model(model_filename)
