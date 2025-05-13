import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,matthews_corrcoef, roc_auc_score, roc_curve,precision_recall_curve, average_precision_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,ExtraTreesClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# suppresses warnings from sklearn and catboost
warnings.filterwarnings("ignore")
# if the results folder does not exist, create it
os.makedirs("results", exist_ok=True)

# loads in the dataset
rawDataFrame = pd.read_csv("BRUIIoT.csv")
# if a missing value is found, drop the row
if rawDataFrame.isnull().values.any():
    rawDataFrame.dropna(inplace=True)

targetColumn = "is_attack__most_frequent"
labelColumns = [targetColumn,"attack_label_enc__most_frequent","attack_label__most_frequent"]
featuresDf = rawDataFrame.drop(labelColumns, axis=1)
targetVector = rawDataFrame[targetColumn].copy()

# 1 = attack, 0 = normal
# if the target vector is not binary, convert it to binary
if targetVector.dtype == object or str(targetVector.dtype).startswith("category"):
    targetVector = LabelEncoder().fit_transform(targetVector)
if np.bincount(targetVector)[0] < np.bincount(targetVector)[1]:
    targetVector = 1 - targetVector

# split the dataset into training and testing sets
trainFeaturesDf, testFeaturesDf, trainLabels, testLabels = train_test_split(
    featuresDf, targetVector, test_size=0.2, stratify=targetVector, random_state=42
)
dataScaler = StandardScaler()
trainFeatures = dataScaler.fit_transform(trainFeaturesDf.values)
testFeatures = dataScaler.transform(testFeaturesDf.values)

print(f"Train/test: {len(trainLabels)} / {len(testLabels)} samples")

# function is used to evaluate the models and print the metrics
def evaluate(model, name, Xtr, ytr, Xte, yte, store):
    ypred = model.predict(Xte)

    if hasattr(model, "predict_proba"):
        yscore = model.predict_proba(Xte)[:,1]
    elif hasattr(model, "decision_function"):
        yscore = model.decision_function(Xte)
    else:
        yscore = None

    # accuracy, precision, recall, f1-score, mcc, roc_auc
    acc = accuracy_score(yte, ypred)
    prec = precision_score(yte, ypred, zero_division=0)
    rec = recall_score(yte, ypred, zero_division=0)
    f1 = f1_score(yte, ypred, zero_division=0)
    mcc = matthews_corrcoef(yte, ypred)
    roc_auc = roc_auc_score(yte, yscore) if yscore is not None else np.nan

    # used to store the metrics for each model
    store.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "MCC": mcc,
        "ROC-AUC": roc_auc
    })

    # ROC aka Receiver Operating Characteristic
    if yscore is not None:
        fpr, tpr, _ = roc_curve(yte, yscore)
        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"{name} ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(loc="lower right")
        plt.savefig(f"results/roc_{name}.png"); plt.close()

        # Precision-Recall
        pre, rec, _ = precision_recall_curve(yte, yscore)
        ap = average_precision_score(yte, yscore)
        base = sum(yte) / len(yte)
        plt.figure(); plt.plot(rec, pre, label=f"AP={ap:.3f}")
        plt.hlines(base, 0, 1, linestyles='--', label="baseline")
        plt.title(f"{name} PR"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.savefig(f"results/pr_{name}.png"); plt.close()

# baseline training for all models
baselineResults = []
baselineModels = {
    "AdaBoost":      AdaBoostClassifier(random_state=42),
    "RandomForest":  RandomForestClassifier(random_state=42),
    "ExtraTrees":    ExtraTreesClassifier(random_state=42),
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "XGBoost":       XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
    "CatBoost":      CatBoostClassifier(verbose=0, random_state=42)
}

# trains models using the default parameters and evaluates them (used to get a baseline for how the models perform)
print("\n=== Baseline ===")
for name, clf in baselineModels.items():
    print(f"\nTraining {name}")
    clf.fit(trainFeatures, trainLabels)
    evaluate(clf, name, trainFeatures, trainLabels, testFeatures, testLabels, baselineResults)

# using 5% of the training data for tuning
subsampleFeatures, _, subsampleLabels, _ = train_test_split(
    trainFeatures, trainLabels, train_size=0.05, stratify=trainLabels, random_state=42
)

# variable to store tuned results
tunedResults = []

                                                # AdaBoost #
#####################################################################################################################
print("\n================== Tuning AdaBoost ==================")
adaBoostParamGrid = {
    "n_estimators":  [100, 300, 500],
    "learning_rate": [0.01, 0.1, 0.5],
    "estimator":     [DecisionTreeClassifier(max_depth=d, random_state=42) for d in (1, 3, 5)]
}
adaBoostGrid = GridSearchCV(AdaBoostClassifier(random_state=42),adaBoostParamGrid,scoring="f1", cv=5, n_jobs=4, verbose=2)
adaBoostGrid.fit(subsampleFeatures, subsampleLabels)
bestAdaBoost = adaBoostGrid.best_estimator_
print("AB params:", adaBoostGrid.best_params_)
evaluate(bestAdaBoost, "AdaBoost_Tuned", trainFeatures, trainLabels, testFeatures, testLabels, tunedResults)

                                                # GradientBoost #
#####################################################################################################################
print("\n================== Tuning GradientBoost ==================")
gradientBoostParamGrid = {
    "n_estimators":  [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth":     [3, 5, 7]
}
gradientBoostGrid = GridSearchCV(GradientBoostingClassifier(random_state=42),gradientBoostParamGrid,scoring="f1", cv=5, n_jobs=4, verbose=2)
gradientBoostGrid.fit(subsampleFeatures, subsampleLabels)
bestGradientBoost = gradientBoostGrid.best_estimator_
print("GB params:", gradientBoostGrid.best_params_)
evaluate(bestGradientBoost, "GradientBoost_Tuned", trainFeatures, trainLabels, testFeatures, testLabels, tunedResults)

                                                # RandomForest #
#####################################################################################################################
print("\n================== Tuning RandomForest ==================")
randomForestParamGrid = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [None, 10, 20],
    "min_samples_leaf": [1, 4, 10]
}
randomForestGrid = GridSearchCV(RandomForestClassifier(random_state=42),randomForestParamGrid,scoring="f1", cv=5, n_jobs=4, verbose=2)
randomForestGrid.fit(subsampleFeatures, subsampleLabels)
bestRandomForest = randomForestGrid.best_estimator_
print("RF params:", randomForestGrid.best_params_)
evaluate(bestRandomForest, "RandomForest_Tuned", trainFeatures, trainLabels, testFeatures, testLabels, tunedResults)

                                                # ExtraTrees #
#####################################################################################################################
print("\n================== Tuning ExtraTrees ==================")
extraTreesParamGrid = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [None, 10, 20],
    "min_samples_leaf": [1, 4, 10],
    "max_features":     ["sqrt", "log2", 0.5]
}
extraTreesGrid = GridSearchCV(ExtraTreesClassifier(random_state=42),extraTreesParamGrid,scoring="f1", cv=5, n_jobs=4, verbose=2)
extraTreesGrid.fit(subsampleFeatures, subsampleLabels)
bestExtraTrees = extraTreesGrid.best_estimator_
print("ET params:", extraTreesGrid.best_params_)
evaluate(bestExtraTrees, "ExtraTrees_Tuned", trainFeatures, trainLabels, testFeatures, testLabels, tunedResults)

# saving all the metrics to a csv file
allResults = pd.DataFrame(baselineResults + tunedResults)
allResults.to_csv("results/model_metrics_full.csv", index=False)
print("\nSaved model_metrics_full.csv with baseline + tuned metrics")
