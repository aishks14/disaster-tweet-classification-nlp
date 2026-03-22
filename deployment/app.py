import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, re, string, time
import plotly.graph_objects as go
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = Path(r"E:/DData/Projects/DSC/NextHikes/Python/disaster-tweet-classification-nlp-pro-7/models")

st.set_page_config(page_title='DisasterScan', page_icon='!',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&display=swap');
:root{--bg:#0a0a0f;--surface:#13131a;--border:#1e1e2e;--accent:#ff3c3c;--safe:#00e5a0;--text:#e8e8f0;--sub:#888899;}
html,body,[class*='css']{font-family:'DM Mono',monospace;background:var(--bg);color:var(--text);}
.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.hero{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;background:linear-gradient(135deg,#ff3c3c,#ff8c42,#ffd166);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.sub{font-size:.8rem;color:var(--sub);letter-spacing:.12em;text-transform:uppercase;margin-top:.2rem;}
.stTextArea textarea{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;font-family:'DM Mono',monospace!important;}
.stTextArea textarea:focus{border-color:var(--accent)!important;}
.stButton>button{font-family:'Syne',sans-serif!important;font-weight:700!important;text-transform:uppercase!important;background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:6px!important;padding:.5rem 1.8rem!important;}
.stButton>button:hover{background:#ff5555!important;}
.card{border-radius:10px;padding:1.5rem 1.8rem;margin:.8rem 0;border:1px solid;}
.dis{background:rgba(255,60,60,.07);border-color:rgba(255,60,60,.3);}
.safe{background:rgba(0,229,160,.07);border-color:rgba(0,229,160,.3);}
.lbl{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;margin:0;}
.lbl-dis{color:var(--accent);} .lbl-safe{color:var(--safe);}
.meta{font-size:.75rem;color:var(--sub);margin-top:.3rem;text-transform:uppercase;letter-spacing:.05em;}
.badge{background:var(--border);border-radius:5px;padding:.4rem .7rem;font-size:.75rem;color:var(--sub);margin-top:.6rem;word-break:break-word;}
.pill{display:inline-block;background:var(--surface);border:1px solid var(--border);border-radius:999px;padding:.2rem .8rem;font-size:.72rem;color:var(--sub);margin:.15rem;}
[data-testid='stSidebar']{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
.stag{font-family:'Syne',sans-serif;font-size:.65rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);border:1px solid rgba(255,60,60,.3);border-radius:4px;padding:.1rem .45rem;display:inline-block;margin-bottom:.5rem;}
.mrow{display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid var(--border);font-size:.78rem;}
.mk{color:var(--sub);} .mv{font-family:'Syne',sans-serif;font-weight:700;}
.stTabs [data-baseweb='tab-list']{background:transparent;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb='tab']{font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;text-transform:uppercase;color:var(--sub);background:transparent;border:none;padding:.45rem .9rem;}
.stTabs [aria-selected='true']{color:var(--text)!important;border-bottom:2px solid var(--accent)!important;}
</style>''', unsafe_allow_html=True)

def clean_text(text):
    text = str(text).lower()
    import re, string
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def confidence_gauge(prob, is_dis):
    col = '#ff3c3c' if is_dis else '#00e5a0'
    val = prob if is_dis else 1 - prob
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=val*100,
        number=dict(suffix=' %', font=dict(color=col, size=34, family='Syne')),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor='#333', tickfont=dict(color='#555', size=9)),
            bar=dict(color=col, thickness=0.55),
            bgcolor='#13131a', bordercolor='#1e1e2e',
            steps=[dict(range=[0,40],color='#1a1a24'),dict(range=[40,70],color='#1e1e2c'),dict(range=[70,100],color='#222230')],
            threshold=dict(line=dict(color=col,width=3),thickness=0.75,value=val*100),
        ),
    ))
    fig.update_layout(height=210, margin=dict(l=15,r=15,t=15,b=8),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#888'))
    return fig

def prob_bar(prob):
    d = round(prob*100, 1)
    s = round((1-prob)*100, 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[d], y=[''], orientation='h', marker_color='#ff3c3c',
        text=f'Disaster: {d}%', textposition='inside', textfont=dict(color='white',size=11)))
    fig.add_trace(go.Bar(x=[s], y=[''], orientation='h', marker_color='#00e5a0',
        text=f'Safe: {s}%', textposition='inside', textfont=dict(color='#0a0a0f',size=11)))
    fig.update_layout(barmode='stack', height=65, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, xaxis=dict(visible=False,range=[0,100]), yaxis=dict(visible=False))
    return fig

@st.cache_resource(show_spinner='Loading model...')
def load_predictor():
    with open(MODEL_DIR / 'model_metadata.json') as f:
        meta = json.load(f)
    tfidf    = pickle.load(open(MODEL_DIR / 'tfidf.pkl', 'rb'))
    lstm_tok = pickle.load(open(MODEL_DIR / 'lstm_tokenizer.pkl', 'rb'))
    is_dl = meta['is_deep_learning']
    model = load_model(MODEL_DIR / 'lstm_model.h5') if is_dl else \
            pickle.load(open(MODEL_DIR / 'best_model.pkl', 'rb'))
    return model, tfidf, lstm_tok, meta

def run_predict(text, model, tfidf, lstm_tok, meta):
    cleaned = clean_text(text)
    max_len = meta['max_len']
    is_dl   = meta['is_deep_learning']
    # Threshold: always use 0.5 — a standard, safe boundary.
    # The metadata best_threshold can be miscalibrated;
    # 0.5 ensures non-disaster tweets are not wrongly flagged.
    THRESHOLD = 0.5
    if is_dl:
        seq   = lstm_tok.texts_to_sequences([cleaned])
        X     = pad_sequences(seq, maxlen=max_len, padding='post')
        prob  = float(model.predict(X, verbose=0).flatten()[0])
        label = int(prob >= THRESHOLD)
    elif hasattr(model, 'predict_proba'):
        prob  = float(model.predict_proba(tfidf.transform([cleaned]))[0, 1])
        label = int(prob >= THRESHOLD)
    else:
        # LinearSVC / models without predict_proba:
        # decision_function scores are raw margins, NOT calibrated
        # probabilities. Sigmoid of a large margin is always ~1.0
        # which makes everything look like Disaster. Use predict()
        # directly — it gives the correct 0/1 label.
        label = int(model.predict(tfidf.transform([cleaned]))[0])
        prob  = float(label)
    conf = prob if label else 1.0 - prob
    return {
        'tweet':       text,
        'cleaned':     cleaned,
        'probability': round(prob, 4),
        'label':       'Disaster' if label else 'Non-Disaster',
        'is_disaster': bool(label),
        'confidence':  round(float(conf), 4),
    }

if 'history' not in st.session_state:
    st.session_state.history = []

model, tfidf, lstm_tok, meta = load_predictor()

with st.sidebar:
    st.markdown('<div class="stag">Model Info</div>', unsafe_allow_html=True)
    rows = [('Model', meta['best_model']), ('F1', meta['f1_score']),
            ('Accuracy', meta['accuracy']), ('Precision', meta['precision']),
            ('Recall', meta['recall']), ('Threshold', meta['best_threshold']),
            ('ROC-AUC', meta.get('roc_auc','N/A'))]
    h = ''.join(f'<div class="mrow"><span class="mk">{k}</span><span class="mv">{v}</span></div>' for k,v in rows)
    st.markdown(h, unsafe_allow_html=True)
    st.divider()
    total = len(st.session_state.history)
    n_dis = sum(1 for x in st.session_state.history if x['is_disaster'])
    sh = (f'<div class="mrow"><span class="mk">Predictions</span><span class="mv">{total}</span></div>'
          f'<div class="mrow"><span class="mk">Disaster</span><span class="mv">{n_dis}</span></div>'
          f'<div class="mrow"><span class="mk">Safe</span><span class="mv">{total-n_dis}</span></div>')
    st.markdown(sh, unsafe_allow_html=True)
    st.divider()
    if st.button('Clear History'):
        st.session_state.history = []
        st.rerun()

st.markdown('<h1 class="hero">DisasterScan</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub">Real-time tweet disaster classification · NLP + Deep Learning</p>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Single Tweet', 'Batch CSV', 'History'])

with tab1:
    st.markdown('#### Analyse a tweet')
    examples = ['Type your own...', 'BREAKING: Wildfire sweeps through California, thousands evacuated!',
                'Earthquake 6.2 strikes Japan coast. Tsunami warning issued.',
                'Flood waters rising in Houston - residents urged to evacuate.',
                'Just had the most amazing coffee this morning, loving life!',
                'I am on fire at the gym today, crushed my personal best!',
                'This film is a total disaster, worst thing I have seen all year.']
    sel = st.selectbox('Try an example or type below:', examples)
    default = '' if sel == examples[0] else sel
    tweet_input = st.text_area('Tweet', value=default, height=110,
                               placeholder='Paste or type a tweet here...',
                               label_visibility='collapsed')
    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        go_btn = st.button('Analyse', use_container_width=True)
    with col_info:
        n = len(tweet_input)
        c = '#ff3c3c' if n > 280 else '#555577'
        st.markdown(f'<span style="font-size:.75rem;color:{c}">{n} chars</span>', unsafe_allow_html=True)
    if go_btn:
        if not tweet_input.strip():
            st.warning('Please enter a tweet first.')
        else:
            with st.spinner('Scanning...'):
                res = run_predict(tweet_input, model, tfidf, lstm_tok, meta)
            st.session_state.history.insert(0, res)
            is_dis = res['is_disaster']
            cc  = 'dis' if is_dis else 'safe'
            lc  = 'lbl-dis' if is_dis else 'lbl-safe'
            ico = '[DISASTER]' if is_dis else '[SAFE]'
            conf_pct = round(res['confidence']*100, 1)
            thr = meta['best_threshold']
            card = (f'<div class="card {cc}">'
                    f'<p class="lbl {lc}">{ico} {res["label"]}</p>'
                    f'<p class="meta">Confidence: {conf_pct}% &nbsp;|&nbsp; Threshold: {thr}</p>'
                    f'<div class="badge">Cleaned: {res["cleaned"]}</div>'
                    f'</div>')
            st.markdown(card, unsafe_allow_html=True)
            cg, cb = st.columns([1, 2])
            with cg:
                st.plotly_chart(confidence_gauge(res['probability'], is_dis),
                                use_container_width=True, config={'displayModeBar': False})
            with cb:
                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown('**Probability split**')
                st.plotly_chart(prob_bar(res['probability']),
                                use_container_width=True, config={'displayModeBar': False})
                dp = res['probability']
                st.markdown(f'<span class="pill">disaster: {dp}</span>'
                            f'<span class="pill">safe: {round(1-dp,4)}</span>',
                            unsafe_allow_html=True)

with tab2:
    st.markdown('#### Batch prediction from CSV')
    st.caption('Upload a CSV with a text column. Add a target column (0/1) to see accuracy.')
    uploaded = st.file_uploader('Upload CSV', type='csv', label_visibility='collapsed')
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error('CSV must have a text column.')
        else:
            st.info(f'{len(df)} tweets loaded')
            if st.button('Run Batch Prediction'):
                prog = st.progress(0)
                preds = []
                rows_list = df['text'].fillna('').tolist()
                for i, row in enumerate(rows_list):
                    preds.append(run_predict(row, model, tfidf, lstm_tok, meta))
                    prog.progress((i+1)/len(rows_list))
                prog.empty()
                out = pd.DataFrame(preds)[['tweet','label','probability','confidence','cleaned']]
                out.columns = ['Tweet','Prediction','Disaster Prob','Confidence','Cleaned']
                nt = len(out)
                nd = out['Prediction'].eq('Disaster').sum()
                c1, c2, c3 = st.columns(3)
                c1.metric('Total', nt)
                c2.metric('Disaster', nd)
                c3.metric('Safe', nt-nd)
                if 'target' in df.columns:
                    pl = [1 if p['is_disaster'] else 0 for p in preds]
                    acc = np.mean(np.array(df['target'].values) == np.array(pl))
                    st.success(f'Accuracy: {acc:.4f}')
                st.dataframe(out, use_container_width=True, height=320)
                st.download_button('Download CSV',
                    out.to_csv(index=False).encode('utf-8'),
                    'predictions.csv', 'text/csv')

with tab3:
    st.markdown('#### Prediction history')
    if not st.session_state.history:
        st.info('No predictions yet. Use the Single Tweet tab first.')
    else:
        hdf = pd.DataFrame(st.session_state.history)[['tweet','label','probability','confidence']]
        hdf.columns = ['Tweet','Prediction','Disaster Prob','Confidence']
        st.dataframe(
            hdf.style.map(lambda v: 'color:#ff3c3c' if v=='Disaster' else 'color:#00e5a0',
                          subset=['Prediction']),
            use_container_width=True, height=400)
