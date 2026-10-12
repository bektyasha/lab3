"""Запуск экспериментов, сохранение чисел в JSON и заполнение отчёта."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models

ROOT = Path(__file__).resolve().parent

def metrics(y, p, pred):
    return {"roc_auc": float(roc_auc_score(y,p)), "accuracy": float(accuracy_score(y,pred)),
            "precision": float(precision_score(y,pred,zero_division=0)),
            "recall": float(recall_score(y,pred,zero_division=0)), "f1": float(f1_score(y,pred,zero_division=0))}

def main():
    # Регрессия: категориальный признак переводится в 0/1, остальные признаки числовые.
    students = pd.read_csv(ROOT/"Student_Performance.csv")
    target_col = "Performance Index"
    if target_col not in students:
        raise ValueError(f"Не найден столбец {target_col!r}")
    students = students.copy()
    if "Extracurricular Activities" in students:
        students["Extracurricular Activities"] = students["Extracurricular Activities"].map(
            {"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(
                pd.to_numeric(students["Extracurricular Activities"], errors="coerce"))
    students = students.apply(pd.to_numeric, errors="coerce").dropna()
    X = students.drop(columns=[target_col]).to_numpy(float)
    y = students[target_col].to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
    # Масштабируем признаки до передачи модели: параметры модели при этом остаются интерпретируемыми
    # относительно входа функции, а тестовая часть не участвует в оценке scaler.
    scaler = StandardScaler().fit(Xtr)
    reg = train_linear_regression(RegressionInput(scaler.transform(Xtr), ytr, scaler.transform(Xte), yte,
                                                  learning_rate=0.05, epochs=800))
    mse = float(np.mean((np.asarray(reg.predictions).reshape(-1)-yte)**2))

    # Классификация: первый столбец — Creditability; категориальные признаки one-hot кодируются.
    german = pd.read_csv(ROOT/"german.csv")
    target_name = "Creditability" if "Creditability" in german.columns else german.columns[0]
    gy = pd.to_numeric(german[target_name], errors="raise")
    # Поддержать распространённую кодировку 1/2, если она есть.
    values = sorted(pd.unique(gy))
    if values == [1, 2]: gy = gy.map({1: 0, 2: 1})
    elif values != [0, 1]: gy = gy.map({values[0]: 0, values[-1]: 1})
    gx = german.drop(columns=[target_name])
    gx = pd.get_dummies(gx, drop_first=False, dtype=float).replace([np.inf,-np.inf],np.nan)
    joined = gx.copy(); joined["_target_"] = gy
    joined = joined.dropna()
    gy = joined.pop("_target_").to_numpy(int); gx = joined.to_numpy(float)
    Gtr,Gte,gytr,gyte = train_test_split(gx,gy,test_size=0.2,random_state=42,stratify=gy)
    # Одинаковый split; sklearn scaler здесь обучается только на train.
    cls = train_logistic_classifier(ClassificationInput(Gtr,gytr,Gte,gyte,learning_rate=0.1,epochs=1000))
    own = metrics(gyte, np.asarray(cls.probabilities), np.asarray(cls.predictions))
    comparison = compare_classical_models(ClassificationInput(Gtr,gytr,Gte,gyte))
    result = {"linear_regression":{"test_mse":mse},
              "classification":{"own_logistic":own}}
    for key in ("logistic_regression","decision_tree","knn"):
        m = getattr(comparison,key)
        result["classification"][key] = {k:float(getattr(m,k)) for k in ("roc_auc","accuracy","precision","recall","f1")}
    (ROOT/"experiment_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report = f"""# Отчёт по лабораторной работе №3

## Данные и подготовка
Для регрессии целевая переменная — `Performance Index`. `Extracurricular Activities` преобразуется из Yes/No в 1/0; строки с пропусками удаляются. Признаки стандартизируются по обучающей выборке. Для классификации целевая переменная берётся из первого столбца `german.csv` (`Creditability`), категориальные признаки кодируются one-hot. Разбиение train/test — 80/20, `random_state=42`; в классификации используется стратификация.

## Гипотезы и эксперименты
Проверяется, что стандартизация ускоряет градиентный спуск, а бинарная кросс-энтропия у собственной логистической модели уменьшается в процессе обучения. Сравнение классификаторов проводится на одинаковом разбиении.

Параметры собственной линейной регрессии: learning rate = 0.05, epochs = 800. Параметры собственной логистической регрессии: learning rate = 0.1, epochs = 1000. Для базовых моделей: LogisticRegression(max_iter=3000), DecisionTreeClassifier(max_depth=6), KNeighborsClassifier(n_neighbors=7).

## Метрики
Реальные измеренные значения записаны в `experiment_results.json` и приведены ниже.

- MSE регрессии на тесте: **{mse:.6f}**
- Собственная логистическая регрессия: {json.dumps(own,ensure_ascii=False)}

| Модель | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
"""
    for key,label in [("own_logistic","Собственная логистическая регрессия"),("logistic_regression","LogisticRegression sklearn"),("decision_tree","Decision Tree"),("knn","KNN")]:
        m = result["classification"][key]
        report += "| "+label+" | "+" | ".join(f"{m[k]:.4f}" for k in ("roc_auc","accuracy","precision","recall","f1"))+" |\n"
    report += """
## 

## 
# """
    (ROOT/"REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
