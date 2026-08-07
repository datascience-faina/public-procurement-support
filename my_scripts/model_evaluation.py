# Model Evaluation


"""

"""

from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {"MAE": mae, "R2": r2}