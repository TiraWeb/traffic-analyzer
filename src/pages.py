import streamlit as st
import pandas as pd
import altair as alt
import time
from datetime import datetime

from utils import FEATURE_NAMES, CATEGORICAL_FEATURES, PROTOCOL_OPTIONS, SERVICE_OPTIONS, FLAG_OPTIONS, generate_test_data
from models import load_models, encode_and_scale, predict_and_store


# Session helpers (moved from main.py)
def ensure_session_defaults():
    if 'latest_attack_type' not in st.session_state:
        st.session_state.latest_attack_type = None
    if 'last_batch_counts' not in st.session_state:
        st.session_state.last_batch_counts = None
    if 'last_manual_result' not in st.session_state:
        st.session_state.last_manual_result = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'recent_attack_types' not in st.session_state:
        st.session_state.recent_attack_types = []
    if 'local_traffic_log' not in st.session_state:
        st.session_state.local_traffic_log = []


def render_app_header():
    if not st.session_state.get("__nids_header_rendered__", False):
        st.title("🛡️ Network Intrusion Detection System")
        st.markdown("---")
        st.session_state["__nids_header_rendered__"] = True


# Small helper to render a single prediction result (used by Manual and other flows)
def render_prediction_result(r):
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

def page_home():
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
        binary_model, multiclass_model, scaler, label_encoders, label_encoder_attack = load_models()
        if binary_model and multiclass_model and scaler and label_encoders and label_encoder_attack:
            st.success("Binary Classifier: Loaded")
            st.success("Multiclass Classifier: Loaded")
            st.success("Feature Scaler: Loaded")
            st.success("Label Encoders: Loaded")
            st.info("System Ready for Analysis")
        else:
            st.error("Models failed to load. Confirm files exist in ../models")


def page_intrusion_detection():
    ensure_session_defaults()
    render_app_header()
    st.header("🔍 Intrusion Detection")

    binary_model, multiclass_model, scaler, label_encoders, label_encoder_attack = load_models()

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
                st.success(f"Test data loaded. Example: duration={test['duration']}, protocol_type={test['protocol_type']}")
                st.rerun()
        with colB:
            if st.button("🧹 Clear Form", key="clear_form_btn"):
                for f in FEATURE_NAMES:
                    st.session_state.pop(f"input_{f}", None)
                st.session_state.last_manual_result = None
                st.success("Form cleared.")

        # Manual KDD feature entry
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

    # Tab 3: Live (simple local generator for reportable "live" samples)
    with tab_live:
        st.subheader("Live samples")
        st.write("Generate a small real HTTP fetch from your laptop and analyze it as a 'live' traffic sample. Useful for demonstration and reports.")

        url = st.text_input("Target URL", value=st.session_state.get('local_gen_url', 'http://example.com'), key='local_gen_url')
        col_a, col_b = st.columns([1, 2])
        with col_a:
            if st.button("Generate local traffic sample (HTTP fetch)", key="gen_local_live_btn"):
                try:
                    import urllib.request
                    t0 = time.time()
                    resp = urllib.request.urlopen(url, timeout=7)
                    data = resp.read()
                    duration = max(0.001, time.time() - t0)
                    feats = {
                        'protocol_type': 'tcp',
                        'service': 'http',
                        'flag': 'SF',
                        'duration': int(round(duration)),
                        'src_bytes': len(data),
                        'dst_bytes': 0,
                        'count': 1,
                        'srv_count': 1
                    }
                    # Ensure all FEATURE_NAMES present
                    for key in FEATURE_NAMES:
                        if key not in feats:
                            feats[key] = 0 if key not in CATEGORICAL_FEATURES else 'other'

                    df_single = pd.DataFrame([feats])
                    X_np = encode_and_scale(df_single, scaler, label_encoders)
                    if X_np is None:
                        st.error("Preprocessing failed for generated sample.")
                    else:
                        results = predict_and_store(X_np, multiclass_model, label_encoder_attack)
                        if results:
                            r = results[0]
                            ts = time.time()
                            entry = {
                                'ts': ts,
                                'url': url,
                                'bytes': len(data),
                                'duration': duration,
                                'result': r
                            }
                            log = st.session_state.get('local_traffic_log', [])
                            log.append(entry)
                            st.session_state.local_traffic_log = log[-20:]

                            st.success("Local traffic sample generated and processed.")
                            render_prediction_result(r)
                            st.subheader('Inferred features (from generated sample)')
                            st.json(feats)
                        else:
                            st.error("Prediction failed for generated sample.")
                except Exception as e:
                    st.error(f"Local traffic generation failed: {e}")


def page_visualizations():
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

    if session_total == 0 and latest_total == 0:
        st.info('This page shows demo data until you run an analysis (upload a CSV or use Manual Input -> Load Test Data).')
