import os
import numpy as np

# Two possible feature orders
FEATURES_11 = [
    'chol', 'stab.glu', 'hdl', 'ratio', 'age', 'height', 'weight',
    'bp.1s', 'bp.1d', 'waist', 'hip'
]

FEATURES_12 = [
    'chol', 'stab.glu', 'hdl', 'ratio', 'age', 'gender', 'height', 'weight',
    'bp.1s', 'bp.1d', 'waist', 'hip'
]

DEFAULTS = {
    'chol': 200.0,
    'stab.glu': 100.0,
    'hdl': 50.0,
    'ratio': 3.0,
    'age': 50.0,
    'gender': 1.0,
    'height': 165.0,
    'weight': 70.0,
    'bp.1s': 120.0,
    'bp.1d': 80.0,
    'waist': 80.0,
    'hip': 90.0,
}


def load_resources(model_candidates=None, scaler_path='scaler.pkl'):
    """Load a model (hybrid preferred) and scaler. Returns (model, scaler).
    If joblib is unavailable or files missing, returns (None, None).
    """
    try:
        import joblib
    except Exception:
        return None, None

    if model_candidates is None:
        model_candidates = ['diabetes_hybrid_model.pkl', 'diabetes_ann_model.pkl', 'diabetes_tab_transformer_model.pkl']

    model = None
    for path in model_candidates:
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                break
        except Exception:
            model = None

    scaler = None
    try:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None

    return model, scaler


def make_prediction(model, scaler, values):
    """Given a loaded model and scaler, predict probability and label.
    `values` should be a list or array of shape (n_features,).
    """
    X = np.array(values, dtype=np.float32).reshape(1, -1)

    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}")
    else:
        X_scaled = X

    if model is None:
        raise RuntimeError('No model loaded')

    # ensemble dict
    if isinstance(model, dict):
        probs = []
        for _, m in model.items():
            if hasattr(m, 'predict_proba'):
                p = np.asarray(m.predict_proba(X_scaled))
                probs.append(p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p.flatten())
            elif hasattr(m, 'predict'):
                probs.append(np.asarray(m.predict(X_scaled)).flatten())
        if len(probs) == 0:
            raise RuntimeError('Ensemble has no usable predictors')
        prob = float(np.mean(np.vstack(probs), axis=0).flatten()[0])
    else:
        if hasattr(model, 'predict_proba'):
            p = np.asarray(model.predict_proba(X_scaled))
            prob = float(p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p.flatten()[0])
        elif hasattr(model, 'predict'):
            p = np.asarray(model.predict(X_scaled)).flatten()
            prob = float(p[0])
        else:
            # Try Keras-style predict
            try:
                p = np.asarray(model.predict(X_scaled)).flatten()
                prob = float(p[0])
            except Exception:
                raise RuntimeError('Model has no predict/probability interface')

    label = 'Diabetic' if prob > 0.5 else 'Non-diabetic'
    return label, prob


def default_input_list(use_gender=True):
    if use_gender:
        return [DEFAULTS[f] for f in FEATURES_12]
    return [DEFAULTS[f] for f in FEATURES_11]
