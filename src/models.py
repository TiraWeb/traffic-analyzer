import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import FEATURE_NAMES, CATEGORICAL_FEATURES


@st.cache_resource
def load_models():
    """Load models and encoders from ../models with caching."""
    base_path = os.path.dirname(__file__)
    models_path = os.path.join(base_path, '..', 'models')

    try:
        binary_model = joblib.load(os.path.join(models_path, 'random_forest_ids (1).pkl'))
        multiclass_model = joblib.load(os.path.join(models_path, 'random_forest_ids_multiclass (1).pkl'))
        scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(models_path, 'label_encoders.pkl'))
        label_encoder_attack = joblib.load(os.path.join(models_path, 'label_encoder_attack.pkl'))
        return binary_model, multiclass_model, scaler, label_encoders, label_encoder_attack
    except Exception as e:
        st.error(f"Error loading models: {e}\nTried path: {os.path.abspath(models_path)}")
        return None, None, None, None, None


def encode_and_scale(df: pd.DataFrame, scaler, label_encoders) -> np.ndarray:
    """Encode categoricals, scale features, and return numpy array.

    Uses the shared FEATURE_NAMES list from utils. Errors are reported to Streamlit.
    """
    try:
        df = df[FEATURE_NAMES].copy()

        if label_encoders is None:
            raise RuntimeError("Label encoders not loaded.")
        # Encode categoricals with unseen-handling
        for col in CATEGORICAL_FEATURES:
            le = label_encoders.get(col)
            if le is None:
                raise RuntimeError(f"Missing label encoder for '{col}'.")
            try:
                df[col] = le.transform(df[col])
            except Exception:
                default_val = le.classes_[0]
                df[col] = df[col].apply(lambda x: default_val if x not in le.classes_ else x)
                df[col] = le.transform(df[col])
                st.warning(f"Unseen value in {col}; using default '{default_val}'.")

        # Ensure all numeric before scaling
        if df.select_dtypes(include=['object']).shape[1] > 0:
            raise RuntimeError("Non-numeric columns remain after encoding.")

        if scaler is None:
            raise RuntimeError("Scaler not loaded.")

        # Prefer passing full DF to scaler when it was fit with names
        try:
            X_scaled = scaler.transform(df)
            X_np = X_scaled if isinstance(X_scaled, np.ndarray) else np.asarray(X_scaled)
        except Exception:
            # Fallback: scale only numeric features
            numeric_features = [c for c in FEATURE_NAMES if c not in CATEGORICAL_FEATURES]
            scaled_numeric = scaler.transform(df[numeric_features].to_numpy())
            df.loc[:, numeric_features] = scaled_numeric
            X_np = df.to_numpy()

        return X_np
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None


def predict_and_store(X_np: np.ndarray, multiclass_model, label_encoder_attack):
    """Run multiclass prediction and update Streamlit session state.

    This mirrors previous behavior in the monolithic app and returns the same
    `results` structure (list of dicts) for rendering.
    """
    if X_np is None:
        return None
    try:
        mc_pred = multiclass_model.predict(X_np)
        mc_prob = multiclass_model.predict_proba(X_np)

        # Determine index of the normal class in the encoder (robust)
        classes = list(label_encoder_attack.classes_)
        if 'normal.' in classes:
            normal_idx = classes.index('normal.')
        elif 'normal' in classes:
            normal_idx = classes.index('normal')
        else:
            normal_idx = None

        # Debug / visibility (internal only)
        top_5_probs = sorted(enumerate(mc_prob[0]), key=lambda x: x[1], reverse=True)[:5]
        top_5_names = [(label_encoder_attack.inverse_transform([i])[0], f'{prob:.1%}') for i, prob in top_5_probs]

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

    results = []
    attack_count = 0

    for i in range(len(mc_pred)):
        probs = mc_prob[i]
        prob_normal = float(probs[normal_idx]) if (normal_idx is not None and normal_idx < len(probs)) else 0.5

        entry = {
            'prob_normal': prob_normal,
            'prob_attack': 1.0 - prob_normal,
            'pred_binary': 0 if prob_normal > 0.5 else 1
        }

        if entry['pred_binary'] == 1:
            # Attack: pick top-3 non-normal candidates
            non_normal = [(idx, p) for idx, p in enumerate(probs) if idx != normal_idx]
            top_attacks = sorted(non_normal, key=lambda x: x[1], reverse=True)[:3]

            # Compute normalization denominator: total probability mass for non-normal classes
            sum_non_normal = sum(p for _, p in non_normal)
            sum_top = sum(p for _, p in top_attacks)

            attack_list = []
            for idx, p in top_attacks:
                name = label_encoder_attack.inverse_transform([idx])[0].strip('.')
                raw_prob = float(p)
                raw_pct_of_attack = (raw_prob / sum_non_normal) if (sum_non_normal > 0) else (raw_prob / sum_top if sum_top > 0 else 0.0)
                pct_of_top = (raw_prob / sum_top) if sum_top > 0 else 1.0 / max(len(top_attacks), 1)

                attack_list.append({
                    'name': name,
                    'raw_prob': raw_prob,
                    'raw_pct_of_attack': raw_pct_of_attack,
                    'pct_of_top': pct_of_top
                })

            attack_top_coverage = (sum_top / sum_non_normal) if (sum_non_normal > 0) else 1.0

            entry['attack_type'] = attack_list[0]['name']
            entry['attack_top'] = attack_list
            entry['attack_top_coverage'] = attack_top_coverage
            attack_count += 1
            st.session_state.latest_attack_type = attack_list[0]['name']
            ra = st.session_state.get('recent_attack_types', [])
            ra.append(attack_list[0]['name'])
            st.session_state.recent_attack_types = ra[-20:]

        results.append(entry)

    st.session_state.last_batch_counts = {
        'total': len(results),
        'attack': attack_count,
        'normal': len(results) - attack_count
    }

    hist_entry = {
        'ts': pd.Timestamp.now(),
        'total': len(results),
        'attack': attack_count,
        'normal': len(results) - attack_count
    }
    hist = st.session_state.get('analysis_history', [])
    hist.append(hist_entry)
    st.session_state.analysis_history = hist[-200:]

    return results
