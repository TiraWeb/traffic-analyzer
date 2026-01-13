import os
import random
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import altair as alt
import time
from datetime import datetime



st.set_page_config(page_title="Network Intrusion Detection System", page_icon="🛡️", layout="wide")

# Constants (moved to src/utils.py)
from utils import FEATURE_NAMES, CATEGORICAL_FEATURES, PROTOCOL_OPTIONS, SERVICE_OPTIONS, FLAG_OPTIONS, generate_test_data
from models import load_models, encode_and_scale, predict_and_store
from pages import page_home, page_intrusion_detection, page_visualizations, ensure_session_defaults


# Model Loading
@st.cache_resource
def _legacy_load_models():
    """Load models and encoders from ../models with caching."""
    import joblib
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
        st.error(f"Error loading models: {e}\\nTried path: {os.path.abspath(models_path)}")
        return None, None, None, None, None


binary_model, multiclass_model, scaler, label_encoders, label_encoder_attack = load_models()


# Session State Helpers
def _legacy_ensure_session_defaults():
    if 'latest_attack_type' not in st.session_state:
        st.session_state.latest_attack_type = None
    if 'last_batch_counts' not in st.session_state:
        # {'total': int, 'normal': int, 'attack': int}
        st.session_state.last_batch_counts = None
    if 'last_manual_result' not in st.session_state:
        # {'label': 'Normal'|'Attack', 'confidence': float}
        st.session_state.last_manual_result = None
    if 'analysis_history' not in st.session_state:
        # list of {'ts': Timestamp, 'total': int, 'normal': int, 'attack': int}
        st.session_state.analysis_history = []
    if 'recent_attack_types' not in st.session_state:
        # list of recent top attack type names
        st.session_state.recent_attack_types = []
    if 'local_traffic_log' not in st.session_state:
        # list of generated local traffic samples for reporting
        st.session_state.local_traffic_log = []


def render_app_header():
    if not st.session_state.get("__nids_header_rendered__", False):
        st.title("🛡️ Network Intrusion Detection System")
        st.markdown("---")
        st.session_state["__nids_header_rendered__"] = True


# Utilities
def _legacy_generate_test_data() -> dict:
    d = {}
    
    # 50% chance to generate attack-like patterns
    is_attack = random.random() < 0.5
    
    if is_attack:
        # Generate attack-like data
        attack_patterns = [
            # Smurf/Neptune DoS pattern - EXTREME values
            {
                'name': 'Smurf/Neptune DoS',
                'duration': 0,
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 511,  # Max value
                'srv_count': 511,  # Max value
                'serror_rate': 1.0,  # 100% errors
                'srv_serror_rate': 1.0,  # 100% service errors
                'same_srv_rate': 1.0,  # All same service
                'dst_host_count': 255,  # Max hosts
                'dst_host_srv_count': 255,  # Max service count
                'protocol_type': 'icmp',  # ICMP for smurf
                'service': 'eco_i',
                'flag': 'SF'
            },


            # Satan/Nmap probe pattern
            {
                'name': 'Satan/Nmap probe',
                'duration': 0,
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 150,
                'srv_count': 25,
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'diff_srv_rate': 1.0,  # 100% different services
                'dst_host_count': 255,  # Scanning many hosts
                'dst_host_same_srv_rate': 0.0,  # No same service
                'dst_host_diff_srv_rate': 1.0,  # All different services
                'protocol_type': 'tcp',
                'service': 'other',
                'flag': 'S0'  # Half-open connections
            },
            # Buffer overflow pattern
            {
                'name': 'Buffer overflow',
                'num_compromised': 100,  # High compromise
                'root_shell': 1,
                'su_attempted': 2,
                'hot': 30,  # Very hot
                'count': 1,
                'srv_count': 1,
                'src_bytes': 1000,
                'dst_bytes': 0,
                'protocol_type': 'tcp',
                'service': 'telnet',
                'flag': 'SF'
            },
            # FTP write attack
            {
                'name': 'FTP write',
                'num_file_creations': 10,
                'num_access_files': 10,
                'logged_in': 1,
                'num_compromised': 5,
                'hot': 15,
                'protocol_type': 'tcp',
                'service': 'ftp',
                'flag': 'SF'
            }
        ]
        
        # Choose pattern: random attack if is_attack else use a generated normal pattern
        if is_attack:
            pattern = random.choice(attack_patterns)
            st.info(f"Generated ATTACK pattern: {pattern.get('name', 'attack')}")
        else:
            pattern = {
                'protocol_type': random.choice(PROTOCOL_OPTIONS),
                'service': random.choice(SERVICE_OPTIONS),
                'flag': 'SF',
                'count': random.randint(1, 10),
                'srv_count': random.randint(1, 10),
                'src_bytes': random.randint(0, 2000),
                'dst_bytes': random.randint(0, 2000),
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'same_srv_rate': 1.0,
                'dst_host_count': 1,
                'dst_host_srv_count': 1
            }
            st.info("Generated NORMAL pattern")
        
        # Set categorical features from pattern (override defaults)
        d['protocol_type'] = pattern.get('protocol_type', random.choice(PROTOCOL_OPTIONS))
        d['service'] = pattern.get('service', random.choice(SERVICE_OPTIONS))
        d['flag'] = pattern.get('flag', random.choice(['S0', 'REJ', 'RSTR']))  # More suspicious flags
        
        # Set all numeric features with pattern overrides
        d['duration'] = pattern.get('duration', random.randint(0, 1000))
        d['src_bytes'] = pattern.get('src_bytes', random.randint(0, 20000))
        d['dst_bytes'] = pattern.get('dst_bytes', random.randint(0, 20000))
        d['land'] = pattern.get('land', random.choice([0, 1]))
        d['wrong_fragment'] = pattern.get('wrong_fragment', random.randint(0, 5))
        d['urgent'] = pattern.get('urgent', random.randint(0, 3))
        d['hot'] = pattern.get('hot', random.randint(0, 10))
        d['num_failed_logins'] = pattern.get('num_failed_logins', random.randint(0, 5))
        d['logged_in'] = pattern.get('logged_in', random.choice([0, 1]))
        d['num_compromised'] = pattern.get('num_compromised', random.randint(0, 100))
        d['root_shell'] = pattern.get('root_shell', random.choice([0, 1]))
        d['su_attempted'] = pattern.get('su_attempted', random.choice([0, 1]))
        d['num_root'] = pattern.get('num_root', random.randint(0, 10))
        d['num_file_creations'] = pattern.get('num_file_creations', random.randint(0, 20))
        d['num_shells'] = pattern.get('num_shells', random.randint(0, 5))
        d['num_access_files'] = pattern.get('num_access_files', random.randint(0, 10))
        d['num_outbound_cmds'] = pattern.get('num_outbound_cmds', random.randint(0, 5))
        d['is_host_login'] = pattern.get('is_host_login', random.choice([0, 1]))
        d['is_guest_login'] = pattern.get('is_guest_login', random.choice([0, 1]))
        d['count'] = pattern.get('count', random.randint(1, 500))
        d['srv_count'] = pattern.get('srv_count', random.randint(1, 500))
        d['serror_rate'] = pattern.get('serror_rate', random.uniform(0, 1))
        d['srv_serror_rate'] = pattern.get('srv_serror_rate', random.uniform(0, 1))
        d['rerror_rate'] = pattern.get('rerror_rate', random.uniform(0, 1))
        d['srv_rerror_rate'] = pattern.get('srv_rerror_rate', random.uniform(0, 1))
        d['same_srv_rate'] = pattern.get('same_srv_rate', random.uniform(0, 1))
        d['diff_srv_rate'] = pattern.get('diff_srv_rate', random.uniform(0, 1))
        d['srv_diff_host_rate'] = pattern.get('srv_diff_host_rate', random.uniform(0, 1))
        d['dst_host_count'] = pattern.get('dst_host_count', random.randint(0, 255))
        d['dst_host_srv_count'] = pattern.get('dst_host_srv_count', random.randint(0, 255))
        d['dst_host_same_srv_rate'] = pattern.get('dst_host_same_srv_rate', random.uniform(0, 1))
        d['dst_host_diff_srv_rate'] = pattern.get('dst_host_diff_srv_rate', random.uniform(0, 1))
        d['dst_host_same_src_port_rate'] = pattern.get('dst_host_same_src_port_rate', random.uniform(0, 1))
        d['dst_host_srv_diff_host_rate'] = pattern.get('dst_host_srv_diff_host_rate', random.uniform(0, 1))
    else:
        st.info("Generated NORMAL pattern")
        # Generate normal-looking data (original random approach)
        d['protocol_type'] = random.choice(PROTOCOL_OPTIONS)
        d['service'] = random.choice(SERVICE_OPTIONS)
        d['flag'] = random.choice(['SF', 'S1'])  # Normal flags
        d['duration'] = random.randint(0, 1000)
        d['src_bytes'] = random.randint(0, 20000)
        d['dst_bytes'] = random.randint(0, 20000)
        d['land'] = 0  # Normal traffic
        d['wrong_fragment'] = random.randint(0, 2)
        d['urgent'] = random.randint(0, 1)
        d['hot'] = random.randint(0, 3)
        d['num_failed_logins'] = 0  # No failed logins for normal
        d['logged_in'] = 1  # Successfully logged in
        d['num_compromised'] = 0
        d['root_shell'] = 0
        d['su_attempted'] = 0
        d['num_root'] = random.randint(0, 2)
        d['num_file_creations'] = random.randint(0, 5)
        d['num_shells'] = random.randint(0, 2)
        d['num_access_files'] = random.randint(0, 3)
        d['num_outbound_cmds'] = 0
        d['is_host_login'] = random.choice([0, 1])
        d['is_guest_login'] = 0  # Not guest login
        d['count'] = random.randint(1, 50)
        d['srv_count'] = random.randint(1, 50)
        d['serror_rate'] = random.uniform(0, 0.2)
        d['srv_serror_rate'] = random.uniform(0, 0.2)
        d['rerror_rate'] = random.uniform(0, 0.2)
        d['srv_rerror_rate'] = random.uniform(0, 0.2)
        d['same_srv_rate'] = random.uniform(0.5, 1.0)
        d['diff_srv_rate'] = random.uniform(0, 0.3)
        d['srv_diff_host_rate'] = random.uniform(0, 0.3)
        d['dst_host_count'] = random.randint(0, 50)
        d['dst_host_srv_count'] = random.randint(0, 50)
        d['dst_host_same_srv_rate'] = random.uniform(0.5, 1.0)
        d['dst_host_diff_srv_rate'] = random.uniform(0, 0.3)
        d['dst_host_same_src_port_rate'] = random.uniform(0.5, 1.0)
        d['dst_host_srv_diff_host_rate'] = random.uniform(0, 0.3)
    
    return d







def _legacy_encode_and_scale(df: pd.DataFrame) -> np.ndarray:
    """Encode categoricals, scale features, and return numpy array.
    Uses DataFrame for scaler when it expects named columns, returns numpy for model.
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


def _legacy_predict_and_store(X_np: np.ndarray):
    """Run multiclass prediction and update session state.

    Decision rule:
    - If P(normal) > 0.50 => classify as Normal
    - Else => classify as Attack and present top-3 non-normal candidates
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

        # Debug / visibility (internal only - no UI noise)
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
                # Probability of this candidate given 'attack' mass
                raw_pct_of_attack = (raw_prob / sum_non_normal) if (sum_non_normal > 0) else (raw_prob / sum_top if sum_top > 0 else 0.0)
                # Relative probability among top-3 candidates (sums to ~100% across chosen top-3)
                pct_of_top = (raw_prob / sum_top) if sum_top > 0 else 1.0 / max(len(top_attacks), 1)

                attack_list.append({
                    'name': name,
                    'raw_prob': raw_prob,
                    'raw_pct_of_attack': raw_pct_of_attack,
                    'pct_of_top': pct_of_top
                })

            # Coverage: how much of the total attack mass the top-3 represent
            attack_top_coverage = (sum_top / sum_non_normal) if (sum_non_normal > 0) else 1.0

            entry['attack_type'] = attack_list[0]['name']
            entry['attack_top'] = attack_list
            entry['attack_top_coverage'] = attack_top_coverage
            attack_count += 1
            # Update latest attack type to top candidate
            st.session_state.latest_attack_type = attack_list[0]['name']
            # Track recent attack top candidates (session-local history)
            ra = st.session_state.get('recent_attack_types', [])
            ra.append(attack_list[0]['name'])
            st.session_state.recent_attack_types = ra[-20:]
        
        results.append(entry)

    st.session_state.last_batch_counts = {
        'total': len(results),
        'attack': attack_count,
        'normal': len(results) - attack_count
    }

    # Append a compact summary to analysis history for visualization
    hist_entry = {
        'ts': pd.Timestamp.now(),
        'total': len(results),
        'attack': attack_count,
        'normal': len(results) - attack_count
    }
    hist = st.session_state.get('analysis_history', [])
    hist.append(hist_entry)
    # Keep a bounded history to avoid memory growth
    st.session_state.analysis_history = hist[-200:]

    return results


# Small helper to render a single prediction result (used by Manual and other flows)
def _legacy_render_prediction_result(r):
    """Render a prediction result dict produced by predict_and_store()."""
    if r['pred_binary'] == 0:
        st.success(f"Normal traffic. Confidence: {r['prob_normal']:.2%}")
    else:
        st.error(f"Intrusion detected. Confidence: {r['prob_attack']:.2%}")

    # Top attack candidates
    if 'attack_top' in r and r.get('attack_top'):
        st.markdown("**Top attack candidates**")
        top_items = r.get('attack_top', [])
        df_top = pd.DataFrame(top_items)[["name", "pct_of_top"]].rename(columns={"name": "Attack", "pct_of_top": "% of top-3"})
        df_top["% of top-3"] = df_top["% of top-3"].apply(lambda x: f"{x:.1%}")
        df_top.index = range(1, len(df_top) + 1)
        st.table(df_top)
        if 'attack_top_coverage' in r:
            st.markdown(f"**Top-3 coverage:** {r['attack_top_coverage']:.1%} of attack probability mass")
        st.markdown(f"**Attack Type:** {r['attack_top'][0]['name']}")
    elif 'attack_type' in r:
        st.markdown(f"**Attack Type:** {r['attack_type']}")


# Pages
def _legacy_page_home():
    ensure_session_defaults()
    render_app_header()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("System Overview")
        st.write("""
        This Network Intrusion Detection System (NIDS) uses machine learning to:
        - Detect network intrusions in real-time
        - Classify different types of attacks
        - Provide automated mitigation responses
        - Visualize security analytics
        """)
        st.subheader("Technical Features")
        st.write("""
        - Binary Classification: Normal vs Attack traffic
        - Multiclass Classification: Specific attack type identification
        - Preprocessing Pipeline: Automated feature scaling and encoding
        - Real-time Analysis: Instant threat detection
        """)
    with col2:
        st.subheader("System Status")
        if binary_model and multiclass_model and scaler and label_encoders and label_encoder_attack:
            st.success("Binary Classifier: Loaded")
            st.success("Multiclass Classifier: Loaded")
            st.success("Feature Scaler: Loaded")
            st.success("Label Encoders: Loaded")
            st.info("System Ready for Analysis")
        else:
            st.error("Models failed to load. Confirm files exist in ../models")


def _legacy_page_intrusion_detection():
    ensure_session_defaults()
    render_app_header()
    st.header("🔍 Intrusion Detection")

    if binary_model is None:
        st.error("Models not loaded.")
        return

    tab_csv, tab_manual, tab_live = st.tabs(["Tab 1: Upload CSV", "Tab 2: Manual Input", "Tab 3: Live"])

    # Tab 1: Upload CSV
    with tab_csv:
        st.write("Upload a CSV containing the 41 KDD features in the exact column order.")
        up = st.file_uploader("CSV File", type=["csv"], key="csv_uploader")
        if up is not None:
            try:
                df_up = pd.read_csv(up)
                missing = [c for c in FEATURE_NAMES if c not in df_up.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    st.dataframe(df_up.head())
                    if st.button("Analyze Uploaded CSV", use_container_width=True, key="analyze_csv_btn"):
                        X_np = encode_and_scale(df_up, scaler, label_encoders)
                        results = predict_and_store(X_np, multiclass_model, label_encoder_attack)
                        if results:
                            st.success(f"Analyzed {len(results)} rows: Normal={st.session_state.last_batch_counts['normal']}, Attack={st.session_state.last_batch_counts['attack']}")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # Tab 2: Manual Input
    with tab_manual:
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("🎲 Load Test Data", key="load_test_data_btn"):
                test = generate_test_data()
                for k, v in test.items():
                    st.session_state[f"input_{k}"] = v
                # Debug: show what was loaded
                st.success(f"Test data loaded. Example: duration={test['duration']}, protocol_type={test['protocol_type']}")
                st.rerun()  # Refresh to show updated values
        with colB:
            if st.button("🧹 Clear Form", key="clear_form_btn"):
                for f in FEATURE_NAMES:
                    st.session_state.pop(f"input_{f}", None)
                st.session_state.last_manual_result = None
                st.success("Form cleared.")

        # Manual KDD feature entry (moved inside the Manual tab)
        st.subheader("Enter KDD Features")
        c1, c2, c3 = st.columns(3)
        data = {}
        with c1:
            data['duration'] = st.number_input('duration', min_value=0, value=int(st.session_state.get('input_duration', 0)))
            data['protocol_type'] = st.selectbox('protocol_type', PROTOCOL_OPTIONS, index=PROTOCOL_OPTIONS.index(st.session_state.get('input_protocol_type', 'tcp')))
            data['service'] = st.selectbox('service', SERVICE_OPTIONS, index=SERVICE_OPTIONS.index(st.session_state.get('input_service', 'http')))
            data['flag'] = st.selectbox('flag', FLAG_OPTIONS, index=FLAG_OPTIONS.index(st.session_state.get('input_flag', 'SF')))
            data['src_bytes'] = st.number_input('src_bytes', min_value=0, value=int(st.session_state.get('input_src_bytes', 0)))
            data['dst_bytes'] = st.number_input('dst_bytes', min_value=0, value=int(st.session_state.get('input_dst_bytes', 0)))
            data['land'] = st.selectbox('land', [0, 1], index=int(st.session_state.get('input_land', 0)))
            data['wrong_fragment'] = st.number_input('wrong_fragment', min_value=0, value=int(st.session_state.get('input_wrong_fragment', 0)))
            data['urgent'] = st.number_input('urgent', min_value=0, value=int(st.session_state.get('input_urgent', 0)))
            data['hot'] = st.number_input('hot', min_value=0, value=int(st.session_state.get('input_hot', 0)))
            data['num_failed_logins'] = st.number_input('num_failed_logins', min_value=0, value=int(st.session_state.get('input_num_failed_logins', 0)))
            data['logged_in'] = st.selectbox('logged_in', [0, 1], index=int(st.session_state.get('input_logged_in', 0)))
            data['num_compromised'] = st.number_input('num_compromised', min_value=0, value=int(st.session_state.get('input_num_compromised', 0)))
            data['root_shell'] = st.selectbox('root_shell', [0, 1], index=int(st.session_state.get('input_root_shell', 0)))
        with c2:
            data['su_attempted'] = st.selectbox('su_attempted', [0, 1], index=int(st.session_state.get('input_su_attempted', 0)))
            data['num_root'] = st.number_input('num_root', min_value=0, value=int(st.session_state.get('input_num_root', 0)))
            data['num_file_creations'] = st.number_input('num_file_creations', min_value=0, value=int(st.session_state.get('input_num_file_creations', 0)))
            data['num_shells'] = st.number_input('num_shells', min_value=0, value=int(st.session_state.get('input_num_shells', 0)))
            data['num_access_files'] = st.number_input('num_access_files', min_value=0, value=int(st.session_state.get('input_num_access_files', 0)))
            data['num_outbound_cmds'] = st.number_input('num_outbound_cmds', min_value=0, value=int(st.session_state.get('input_num_outbound_cmds', 0)))
            data['is_host_login'] = st.selectbox('is_host_login', [0, 1], index=int(st.session_state.get('input_is_host_login', 0)))
            data['is_guest_login'] = st.selectbox('is_guest_login', [0, 1], index=int(st.session_state.get('input_is_guest_login', 0)))
            data['count'] = st.number_input('count', min_value=1, value=int(st.session_state.get('input_count', 1)))
            data['srv_count'] = st.number_input('srv_count', min_value=1, value=int(st.session_state.get('input_srv_count', 1)))
            data['serror_rate'] = st.number_input('serror_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_serror_rate', 0.0)))
            data['srv_serror_rate'] = st.number_input('srv_serror_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_srv_serror_rate', 0.0)))
            data['rerror_rate'] = st.number_input('rerror_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_rerror_rate', 0.0)))
            data['srv_rerror_rate'] = st.number_input('srv_rerror_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_srv_rerror_rate', 0.0)))
        with c3:
            data['same_srv_rate'] = st.number_input('same_srv_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_same_srv_rate', 0.0)))
            data['diff_srv_rate'] = st.number_input('diff_srv_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_diff_srv_rate', 0.0)))
            data['srv_diff_host_rate'] = st.number_input('srv_diff_host_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_srv_diff_host_rate', 0.0)))
            data['dst_host_count'] = st.number_input('dst_host_count', min_value=0, max_value=255, value=int(st.session_state.get('input_dst_host_count', 0)))
            data['dst_host_srv_count'] = st.number_input('dst_host_srv_count', min_value=0, max_value=255, value=int(st.session_state.get('input_dst_host_srv_count', 0)))
            data['dst_host_same_srv_rate'] = st.number_input('dst_host_same_srv_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_dst_host_same_srv_rate', 0.0)))
            data['dst_host_diff_srv_rate'] = st.number_input('dst_host_diff_srv_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_dst_host_diff_srv_rate', 0.0)))
            data['dst_host_same_src_port_rate'] = st.number_input('dst_host_same_src_port_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_dst_host_same_src_port_rate', 0.0)))
            data['dst_host_srv_diff_host_rate'] = st.number_input('dst_host_srv_diff_host_rate', min_value=0.0, max_value=1.0, value=float(st.session_state.get('input_dst_host_srv_diff_host_rate', 0.0)))

        st.info("Moved: use Tab 3 — Live to generate and view local traffic samples for reporting.")

        if st.button("🔍 Analyze Traffic", use_container_width=True, key="analyze_traffic_manual_btn"):
            df_single = pd.DataFrame([data])
            X_np = encode_and_scale(df_single, scaler, label_encoders)
            results = predict_and_store(X_np, multiclass_model, label_encoder_attack)
            if results:
                r = results[0]
                if r['pred_binary'] == 0:
                    st.session_state.last_manual_result = {'label': 'Normal', 'confidence': r['prob_normal']}
                    st.success(f"Normal traffic. Confidence: {r['prob_normal']:.2%}")
                else:
                    st.session_state.last_manual_result = {'label': 'Attack', 'confidence': r['prob_attack']}
                    st.error(f"Intrusion detected. Confidence: {r['prob_attack']:.2%}")
                    # Show top-3 attack candidates when available
                    if 'attack_top' in r:
                        st.markdown("### Top attack candidates:")
                        # Render top candidates as a table for clearer presentation
                        top_items = r.get('attack_top', [])
                        if top_items:
                            df_top = pd.DataFrame(top_items)
                            # Keep only expected columns and rename for display
                            df_top = df_top[["name", "pct_of_top"]].rename(
                                columns={"name": "Attack", "pct_of_top": "% of top-3"}
                            )
                            # Format percentage column for readability
                            df_top["% of top-3"] = df_top["% of top-3"].apply(lambda x: f"{x:.1%}")
                            df_top.index = range(1, len(df_top) + 1)
                            st.table(df_top)
                        else:
                            st.write("No candidates available.")

                            # Show coverage of top-3 (nicely highlighted)
                            if 'attack_top_coverage' in r:
                                st.markdown(f"**Top-3 coverage:** {r['attack_top_coverage']:.1%} of attack probability mass")

                            # Also highlight the top candidate
                            if top_items:
                                st.markdown(f"### Attack Type: **{r['attack_top'][0]['name']}**")
                            elif 'attack_type' in r:
                                st.markdown(f"### Attack Type: **{r['attack_type']}**")

def _legacy_page_visualizations():
    ensure_session_defaults()
    render_app_header()
    st.header("📊 Visualizations")

    batch = st.session_state.last_batch_counts
    single = st.session_state.last_manual_result
    history = st.session_state.analysis_history

    # Session aggregate (sum over history)
    if history:
        session_attack = sum(item.get('attack', 0) for item in history)
        session_normal = sum(item.get('normal', 0) for item in history)
        session_total = session_attack + session_normal
    else:
        session_attack = session_normal = session_total = 0

    # Latest batch
    if batch and batch.get('total', 0) > 0:
        latest_total = int(batch.get('total', 0))
        latest_attack = int(batch.get('attack', 0))
        latest_normal = int(batch.get('normal', 0))
    else:
        latest_total = latest_attack = latest_normal = 0

    # Determine which dataset to use for the primary metrics (prefer session aggregate)
    if session_total > 0:
        main_total = session_total
        main_normal = session_normal
        main_attack = session_attack
        subtitle = "Session Aggregate"
    elif latest_total > 0:
        main_total = latest_total
        main_normal = latest_normal
        main_attack = latest_attack
        subtitle = "Latest Batch"
    else:
        # Demo
        main_total = 100
        main_normal = 80
        main_attack = 20
        subtitle = "Demo Data — no analyses yet"

    # Metrics row (show session-level totals primarily)
    normal_pct = (main_normal / main_total) if main_total > 0 else 0.0
    attack_pct = (main_attack / main_total) if main_total > 0 else 0.0

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.metric("Total analyzed", f"{main_total}")
    c2.metric("Normal", f"{main_normal}", f"{normal_pct:.1%}")
    c3.metric("Attack", f"{main_attack}", f"{attack_pct:.1%}")
    if single:
        c4.metric("Last Result", single['label'], f"Confidence {single['confidence']:.1%}")
    else:
        c4.metric("Last Result", "—")

    st.markdown(f"**{subtitle}**")

    # Two-column layout: Left = Session overview (or demo), Right = Latest batch (if present)
    left, right = st.columns([2, 1])

    # Session / demo chart
    with left:
        if session_total > 0:
            df_session = pd.DataFrame({'label': ['Normal', 'Attack'], 'count': [session_normal, session_attack]})
            st.subheader('Session Overview')
        elif latest_total > 0:
            df_session = pd.DataFrame({'label': ['Normal', 'Attack'], 'count': [latest_normal, latest_attack]})
            st.subheader('Latest Overview')
        else:
            df_session = pd.DataFrame({'label': ['Normal', 'Attack'], 'count': [80, 20]})
            st.subheader('Demo Overview')

        df_session['pct'] = df_session['count'] / df_session['count'].sum()
        pie_s = alt.Chart(df_session).mark_arc(innerRadius=70).encode(
            theta=alt.Theta(field='count', type='quantitative'),
            color=alt.Color('label:N', scale=alt.Scale(domain=['Normal', 'Attack'], range=['#4caf50', '#f44336'])),
            tooltip=[alt.Tooltip('label:N', title='Type'), alt.Tooltip('count:Q', title='Count'), alt.Tooltip('pct:Q', format='.1%', title='Percent')]
        ).properties(height=320)
        st.altair_chart(pie_s, use_container_width=True)

        bar_s = alt.Chart(df_session).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X('label:N', title='Type'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('label:N', legend=None, scale=alt.Scale(domain=['Normal', 'Attack'], range=['#4caf50', '#f44336'])),
            tooltip=[alt.Tooltip('label:N', title='Type'), alt.Tooltip('count:Q', title='Count')]
        ).properties(height=120)
        st.altair_chart(bar_s, use_container_width=True)

    # Latest batch column
    with right:
        if latest_total > 0:
            st.subheader('Latest Batch')
            df_latest = pd.DataFrame({'label': ['Normal', 'Attack'], 'count': [latest_normal, latest_attack]})
            df_latest['pct'] = df_latest['count'] / df_latest['count'].sum()
            pie_l = alt.Chart(df_latest).mark_arc(innerRadius=40).encode(
                theta=alt.Theta(field='count', type='quantitative'),
                color=alt.Color('label:N', scale=alt.Scale(domain=['Normal', 'Attack'], range=['#4caf50', '#f44336'])),
                tooltip=[alt.Tooltip('label:N', title='Type'), alt.Tooltip('count:Q', title='Count'), alt.Tooltip('pct:Q', format='.1%', title='Percent')]
            ).properties(height=200)
            st.altair_chart(pie_l, use_container_width=True)
            st.write(f"Total: {latest_total} — Normal: {latest_normal}, Attack: {latest_attack}")
        else:
            st.info('No recent batch analysis. Upload a CSV and run analysis or use Load Test Data -> Analyze Traffic.')

    # History line chart (if present)
    if history:
        df_hist = pd.DataFrame(history)
        df_hist['ts'] = pd.to_datetime(df_hist['ts'])
        df_hist = df_hist.rename(columns={'normal': 'Normal', 'attack': 'Attack'})
        df_long = df_hist.melt(id_vars=['ts'], value_vars=['Normal', 'Attack'], var_name='type', value_name='count')
        line = alt.Chart(df_long).mark_line(point=True).encode(
            x=alt.X('ts:T', title='Time'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('type:N', scale=alt.Scale(domain=['Normal', 'Attack'], range=['#4caf50', '#f44336'])),
            tooltip=[alt.Tooltip('ts:T', title='Time'), alt.Tooltip('type:N', title='Type'), alt.Tooltip('count:Q', title='Count')]
        ).properties(height=220)
        st.subheader('History')
        st.altair_chart(line, use_container_width=True)

    # Recent attack types table
    if st.session_state.recent_attack_types:
        st.subheader('Recent top attack types')
        df_recent = pd.DataFrame({'attack_type': st.session_state.recent_attack_types})
        st.table(df_recent.value_counts().rename_axis('attack_type').reset_index(name='occurrences').sort_values('occurrences', ascending=False).head(10))

    # Helpful note
    if session_total == 0 and latest_total == 0:
        st.info('This page shows demo data until you run an analysis (upload a CSV or use Manual Input -> Load Test Data).')


# Main
def main():
    ensure_session_defaults()

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Intrusion Detection", "Visualizations"],
        index=0,
        key="navigation_select_page"
    )

    if page == "Home":
        page_home()
    elif page == "Intrusion Detection":
        page_intrusion_detection()
    elif page == "Visualizations":
        page_visualizations()


if __name__ == "__main__":
    main()