from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer
)


# 1. Train Random Forest with GridSearchCV
def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100],
        "max_depth": [10],
        "min_samples_split": [2]
        # "min_samples_leaf": [1, 2],
        # "criterion": ["gini", "entropy"]
    }

    # ✅ scorer below param_grid
    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=0   # 👈 same as video
    )

    grid_search.fit(X_train, y_train)

    return grid_search


# 2. Evaluate Classifier (VIDEO STYLE)
def evaluate_classifier(model, X_test, y_test, model_name):

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")   # 👈 formatted like video
    print(report)

    # (optional but better)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


# 3. Print Best Params
def print_best_params(model):
    print("\nBest Parameters:")
    print(model.best_params_)