import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve

dataset = arff.loadarff('speeddating.arff')
df = pd.DataFrame(dataset[0]).apply(lambda x: x.str.decode('utf-8') if x.dtype == "object" else x)

df = df.drop(columns=['expected_num_interested_in_me', 'expected_num_matches'])

cols_to_fix = ['age', 'age_o', 'sports', 'reading', 'yoga'] 
for col in cols_to_fix:
    df[col] = df[col].fillna(df[col].median())

df['funny_o'] = df['funny_o'].fillna(df['funny_o'].mode()[0])

X = df.drop(columns=['match', 'decision', 'decision_o'])
y = df['match'].astype(int)

X = pd.get_dummies(X, drop_first=True)

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_probs):.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Balanced Model)")
plt.savefig('confusion_matrix_final.png')

importances = model.feature_importances_
feat_names = X_imputed.columns
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(10, 8))
plt.title("Top 15 Predictors of a Second Date")
plt.barh(range(len(indices)), importances[indices], color='lightcoral', align='center')
plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc_score(y_test, y_probs):.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

print("\n--- Project Complete ---")
print("Saved: confusion_matrix_final.png, feature_importance.png, and roc_curve.png")