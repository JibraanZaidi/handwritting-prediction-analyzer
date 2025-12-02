# src/train_model.py - Train improved RandomForest models for each trait
import pandas as pd, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
TRAITS = ['extrovert','neurotic','open','agree','conscientious']

DATA_FEATURES_CSV = 'data/features.csv'
MODEL_DIR = 'models'

def train_all(csv_path=DATA_FEATURES_CSV, model_dir=MODEL_DIR):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ['filename'] + TRAITS]
    X = df[feature_cols]
    for trait in TRAITS:
        y = df[trait].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print('Trait:', trait)
        print(classification_report(y_test, preds))
        joblib.dump({'model': clf, 'features': feature_cols, 'trait': trait}, os.path.join(model_dir, f'rf_{trait}.pkl'))
        print('Saved model for', trait)

if __name__ == '__main__':
    if not os.path.exists(DATA_FEATURES_CSV):
        print('Run src/dataset_prep.py first to create data/features.csv')
    else:
        train_all()
