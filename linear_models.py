"""Реализация моделей для лабораторной работы №3."""
import numpy as np
from grader_contracts.linear_models import (
    BaselineComparisonResult, ClassificationInput, ClassificationResult,
    RegressionInput, RegressionResult, ClassificationMetrics,
)

_EPS = 1e-12

def _xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("Признаки должны быть двумерными, а число строк совпадать с target.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Перед обучением обработайте пропуски и бесконечные значения.")
    return x, y

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Собственная линейная регрессия с пакетным градиентным спуском."""
    x, y = _xy(data.train_features, data.train_target)
    xt, _ = _xy(data.test_features, data.test_target)
    if x.shape[1] != xt.shape[1]:
        raise ValueError("Train/test должны иметь одинаковое число признаков.")
    xm, xs = x.mean(axis=0), x.std(axis=0)
    xs[xs < _EPS] = 1.0
    ym, ys = float(y.mean()), float(y.std())
    if ys < _EPS: ys = 1.0
    z = (x-xm)/xs
    yn = (y-ym)/ys
    w = np.zeros(z.shape[1], dtype=float)
    b = 0.0
    lr = float(data.learning_rate)
    history = []
    for _ in range(int(data.epochs)):
        err = z @ w + b - yn
        history.append(float(np.mean(err**2)))
        # Пакетные градиенты по всем обучающим объектам
        dw = (2.0 / len(z)) * (z.T @ err)
        db = 2.0 * float(err.mean())
        w -= lr * dw
        b -= lr * db
        if not np.isfinite(w).all() or not np.isfinite(b):
            raise FloatingPointError("Градиентный спуск разошёлся; уменьшите learning_rate.")
    # Перевод параметров из стандартизированного пространства в исходные единицы.
    weights = ys * w / xs
    bias = ym + ys*b - float(xm @ weights)
    pred = np.asarray(data.test_features, dtype=float)
    if pred.ndim == 1: pred = pred.reshape(-1, 1)
    predictions = pred @ weights + bias
    # История MSE возвращается в единицах исходной целевой переменной.
    loss_history = [float(v * ys**2) for v in history]
    return RegressionResult(weights=weights, bias=float(bias),
                            loss_history=loss_history, predictions=predictions)

def _sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Собственная бинарная логистическая регрессия с пакетным градиентным спуском."""
    x, y = _xy(data.train_features, data.train_target)
    xt, _ = _xy(data.test_features, data.test_target)
    if x.shape[1] != xt.shape[1]:
        raise ValueError("Train/test должны иметь одинаковое число признаков.")
    labels = np.unique(y)
    if not set(labels.tolist()).issubset({0, 1}) or len(labels) != 2:
        raise ValueError("Для бинарной логистической регрессии target должен содержать метки 0 и 1.")
    xm, xs = x.mean(axis=0), x.std(axis=0)
    xs[xs < _EPS] = 1.0
    z, zt = (x-xm)/xs, (xt-xm)/xs
    w, b = np.zeros(z.shape[1]), 0.0
    lr = float(data.learning_rate)
    history = []
    for _ in range(int(data.epochs)):
        p = np.clip(_sigmoid(z @ w + b), _EPS, 1.0-_EPS)
        history.append(float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))))
        err = p-y
        w -= lr * (z.T @ err) / len(z)
        b -= lr * float(err.mean())
        if not np.isfinite(w).all() or not np.isfinite(b):
            raise FloatingPointError("Градиентный спуск разошёлся; уменьшите learning_rate.")
    probs = _sigmoid(zt @ w + b)
    return ClassificationResult(weights=w, bias=float(b), loss_history=history,
                                probabilities=probs, predictions=(probs >= 0.5).astype(int))

def _metrics(y, probabilities, predictions):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    y = np.asarray(y).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    pred = np.asarray(predictions).reshape(-1)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y, p)),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнение sklearn LogisticRegression, дерева и KNN на переданном split."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    x, y = _xy(data.train_features, data.train_target)
    xt, yt = _xy(data.test_features, data.test_target)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    }
    results = {}
    for name, model in models.items():
        model.fit(x, y.astype(int))
        probabilities = model.predict_proba(xt)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results[name] = _metrics(yt, probabilities, predictions)
    return BaselineComparisonResult(**results)
