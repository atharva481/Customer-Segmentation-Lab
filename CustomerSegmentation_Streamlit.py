import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import pickle
import streamlit as st
import os
from datetime import datetime, date
import squarify
import base64
from dotenv import load_dotenv

# ─── Config ───────────────────────────────────────────────────────────────────
load_dotenv()
FEEDBACK_PATH = os.getenv('FEEDBACK_PATH', 'feedback.csv')
DATA_PATH     = os.getenv('DATA_PATH', 'data')

st.set_page_config(
    page_title="Customer Segmentation Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
}
.stMetric {
    background: #f8f9ff;
    border: 1px solid #e0e4ff;
    border-radius: 10px;
    padding: 12px;
}
.feature-badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
div[data-testid="stSidebar"] {
    background: #0f0f1a;
    color: #fff;
}
div[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("🔬 Customer Segmentation Lab")
st.caption("Feature Engineering & Machine Learning Pipeline")

# ─── Menu ─────────────────────────────────────────────────────────────────────
menu = [
    "📋 Business Understanding",
    "📊 Data Understanding",
    "⚙️  Data Preparation",
    "🧪 Feature Engineering",       # NEW
    "📈 Advanced EDA",               # NEW
    "🤖 Modeling & Evaluation",
    "🏆 Model Comparison",           # NEW
    "🎯 Cluster Profiling",          # NEW
    "🔮 Predict"
]
choice = st.sidebar.selectbox("Navigation", menu)

# ─── Session State Init ───────────────────────────────────────────────────────
for key in ['df', 'uploaded_file', 'df_RFM', 'df_RFM_engineered',
            'best_model', 'best_model_name', 'cluster_stats', 'scaler']:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_data(uploaded_file):
    if uploaded_file is None:
        st.warning("Please upload a data file to proceed.")
        return None
    df = pd.read_csv(
        uploaded_file, encoding='latin-1', sep=r'\s+',
        header=None, names=['Customer_id', 'day', 'Quantity', 'Sales']
    )
    df['day'] = pd.to_datetime(df['day'], format='%Y%m%d')
    st.session_state['df'] = df
    st.session_state['reference_date'] = df['day'].max()
    return df

def csv_download_link(df, filename, label):
    b64 = base64.b64encode(df.to_csv(index=True).encode()).decode()
    st.markdown(
        f'<a href="data:file/csv;base64,{b64}" download="{filename}">'
        f'⬇️ {label}</a>', unsafe_allow_html=True
    )

def save_feedback(text):
    fb = pd.DataFrame({'Time': [datetime.now()], 'Feedback': [text]})
    if not os.path.isfile(FEEDBACK_PATH):
        fb.to_csv(FEEDBACK_PATH, index=False)
    else:
        fb.to_csv(FEEDBACK_PATH, mode='a', header=False, index=False)

def compute_rfm(df):
    recent_date = df['day'].max()
    rfm = df.groupby('Customer_id').agg(
        Recency   = ('day', lambda x: (recent_date - x.max()).days),
        Frequency = ('Customer_id', 'count'),
        Monetary  = ('Sales', 'sum')
    )
    return rfm

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BUSINESS UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════════════════
if choice == "📋 Business Understanding":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Business Objective")
        st.markdown("""
        Customer segmentation divides a customer base into groups that share similar characteristics.
        This lab goes beyond basic clustering — it applies **feature engineering** to extract richer
        signals from raw transactional data.

        **Goals:**
        - 🎯 **Personalization** — tailor strategies per segment
        - 💰 **Optimization** — efficient marketing spend
        - 📡 **Insight** — deeper understanding of purchase behaviour
        - 🔄 **Retention** — identify at-risk customers early

        **Feature Engineering Focus (Lab):**
        - RFM base features (Recency, Frequency, Monetary)
        - Derived features: AOV, purchase velocity, tenure
        - Log / sqrt / robust transforms to reduce skew
        - Outlier detection & removal
        - Scaling strategies comparison

        **Problem:** Use Python ML techniques to segment customers and evaluate
        which feature engineering pipeline yields the most meaningful clusters.
        """)
    with col2:
        st.info("**Pipeline Steps**\n\n"
                "1. Load raw transactions\n"
                "2. Inspect & clean data\n"
                "3. Engineer RFM + derived features\n"
                "4. Advanced EDA\n"
                "5. Train & compare models\n"
                "6. Profile clusters\n"
                "7. Predict new customers")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "📊 Data Understanding":
    data_source = st.sidebar.radio('Data source', ['Sample file', 'Upload file'])

    if data_source == 'Sample file' and os.path.exists(DATA_PATH):
        files = os.listdir(DATA_PATH)
        sel = st.sidebar.selectbox('Choose sample file', files)
        st.session_state['uploaded_file'] = open(os.path.join(DATA_PATH, sel), 'r')
        load_data(st.session_state['uploaded_file'])
    else:
        up = st.sidebar.file_uploader("Choose a .txt file", type=['txt'])
        if up:
            st.session_state['uploaded_file'] = up
            load_data(up)

    df = st.session_state['df']
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Unique Customers", f"{df['Customer_id'].nunique():,}")
        c4.metric("Date Range", f"{(df['day'].max()-df['day'].min()).days} days")

        st.write("#### Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        st.write("#### Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        # Sales trend
        st.write("#### Monthly Sales Trend")
        monthly = df.copy()
        monthly['month'] = monthly['day'].dt.to_period('M').dt.to_timestamp()
        fig = px.line(
            monthly.groupby('month')['Sales'].sum().reset_index(),
            x='month', y='Sales', markers=True,
            template='plotly_white', title='Total Revenue per Month'
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "⚙️  Data Preparation":
    df = st.session_state['df']
    if df is None:
        st.warning("Load data in **Data Understanding** first.")
    else:
        st.subheader("Data Quality Report")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Missing Values**")
            st.dataframe(df.isnull().sum().rename("Missing"))
        with col2:
            st.write("**Duplicate Rows**")
            st.metric("Count", df.duplicated().sum())

        if st.checkbox("✅ Remove duplicates"):
            df.drop_duplicates(inplace=True)
            st.session_state['df'] = df
            st.success(f"Removed duplicates. Remaining rows: {len(df):,}")

        if st.checkbox("✅ Remove rows with missing Customer_id"):
            df.dropna(subset=['Customer_id'], inplace=True)
            st.session_state['df'] = df
            st.success(f"Removed nulls. Remaining rows: {len(df):,}")

        if st.checkbox("✅ Remove negative Sales / Quantity (returns/errors)"):
            df = df[(df['Sales'] > 0) & (df['Quantity'] > 0)]
            st.session_state['df'] = df
            st.success(f"Cleaned. Remaining rows: {len(df):,}")

        st.write("#### Distribution of Numeric Columns")
        num_cols = df.select_dtypes(include='number').columns.tolist()
        fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
        for ax, col in zip(axes, num_cols):
            df[col].hist(ax=ax, bins=40, color='#667eea', edgecolor='white')
            ax.set_title(col)
            ax.set_xlabel('')
        plt.tight_layout()
        st.pyplot(fig)

        # Compute base RFM and store
        rfm = compute_rfm(df)
        st.session_state['df_RFM'] = rfm
        st.success("✅ RFM base features computed and saved to session.")
        st.dataframe(rfm.head())

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING  ★ NEW ★
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "🧪 Feature Engineering":
    df    = st.session_state['df']
    df_RFM = st.session_state['df_RFM']

    if df is None or df_RFM is None:
        st.warning("Complete **Data Preparation** first.")
    else:
        st.subheader("🧪 Feature Engineering Pipeline")

        # ── Derived Features ──────────────────────────────────────────────────
        st.write("### Step 1 — Derive New Features")

        eng = df_RFM.copy()

        # Average Order Value
        revenue = df.groupby('Customer_id')['Sales'].sum()
        freq    = df.groupby('Customer_id')['Sales'].count()
        eng['AOV'] = (revenue / freq).round(2)          # Average Order Value

        # Purchase Velocity (orders per week)
        tenure_days = df.groupby('Customer_id')['day'].apply(
            lambda x: max((x.max() - x.min()).days, 1)
        )
        eng['Tenure_Days']         = tenure_days
        eng['Purchase_Velocity']   = (eng['Frequency'] / (eng['Tenure_Days'] / 7)).round(4)

        # Days since first purchase
        first_purchase = df.groupby('Customer_id')['day'].min()
        recent_date    = df['day'].max()
        eng['Days_Since_First']    = (recent_date - first_purchase).dt.days

        # Revenue Consistency (std of monthly spend)
        df['month'] = df['day'].dt.to_period('M')
        monthly_rev = df.groupby(['Customer_id', 'month'])['Sales'].sum().reset_index()
        rev_std     = monthly_rev.groupby('Customer_id')['Sales'].std().fillna(0)
        eng['Revenue_Std']         = rev_std

        st.write("Derived features added:")
        st.dataframe(eng.head(10), use_container_width=True)

        st.markdown("---")
        # ── Outlier Detection ────────────────────────────────────────────────
        st.write("### Step 2 — Outlier Detection & Removal")

        for feat in ['Recency','Frequency','Monetary','AOV']:
            q1, q3 = eng[feat].quantile(0.25), eng[feat].quantile(0.75)
            iqr = q3 - q1
            outliers = ((eng[feat] < q1 - 1.5*iqr) | (eng[feat] > q3 + 1.5*iqr)).sum()
            st.write(f"**{feat}**: {outliers} outliers detected")

        remove_outliers = st.checkbox("✅ Remove outliers (IQR method) from RFM features")
        if remove_outliers:
            for feat in ['Recency','Frequency','Monetary']:
                q1, q3 = eng[feat].quantile(0.25), eng[feat].quantile(0.75)
                iqr = q3 - q1
                eng = eng[(eng[feat] >= q1 - 1.5*iqr) & (eng[feat] <= q3 + 1.5*iqr)]
            st.success(f"After outlier removal: {len(eng):,} customers remain")

        st.markdown("---")
        # ── Transformations ──────────────────────────────────────────────────
        st.write("### Step 3 — Skewness & Transformations")

        skew_table = pd.DataFrame({
            'Feature': eng.select_dtypes('number').columns,
            'Skewness': [eng[c].skew() for c in eng.select_dtypes('number').columns]
        }).sort_values('Skewness', ascending=False)
        st.dataframe(skew_table, use_container_width=True)

        transform = st.selectbox(
            "Apply transform to skewed features (|skew| > 1):",
            ["None", "Log1p", "Sqrt", "Box-Cox (approx)"]
        )
        skewed_cols = skew_table[skew_table['Skewness'].abs() > 1]['Feature'].tolist()

        if transform == "Log1p":
            for c in skewed_cols:
                if eng[c].min() >= 0:
                    eng[c] = np.log1p(eng[c])
            st.success(f"Log1p applied to: {skewed_cols}")
        elif transform == "Sqrt":
            for c in skewed_cols:
                if eng[c].min() >= 0:
                    eng[c] = np.sqrt(eng[c])
            st.success(f"Sqrt applied to: {skewed_cols}")
        elif transform == "Box-Cox (approx)":
            from scipy.stats import boxcox
            for c in skewed_cols:
                if eng[c].min() > 0:
                    eng[c], _ = boxcox(eng[c])
            st.success(f"Box-Cox applied to: {skewed_cols}")

        st.markdown("---")
        # ── Scaling ──────────────────────────────────────────────────────────
        st.write("### Step 4 — Feature Scaling")
        scaler_choice = st.selectbox(
            "Select scaler:",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"]
        )
        feature_cols = st.multiselect(
            "Features to include in model:",
            options=eng.select_dtypes('number').columns.tolist(),
            default=['Recency', 'Frequency', 'Monetary']
        )

        if st.button("✅ Apply Engineering Pipeline"):
            scaler_map = {
                "StandardScaler": StandardScaler(),
                "MinMaxScaler":   MinMaxScaler(),
                "RobustScaler":   RobustScaler()
            }
            scaler = scaler_map[scaler_choice]
            eng_scaled = eng[feature_cols].copy()
            eng_scaled[feature_cols] = scaler.fit_transform(eng_scaled[feature_cols])

            st.session_state['df_RFM_engineered'] = eng_scaled
            st.session_state['df_RFM_raw_eng']    = eng      # unscaled, for profiling
            st.session_state['scaler']             = scaler
            st.session_state['feature_cols']       = feature_cols

            st.success(f"✅ Pipeline complete! {len(eng_scaled):,} customers, {len(feature_cols)} features → ready for modeling.")
            st.dataframe(eng_scaled.head(), use_container_width=True)

            # Correlation heatmap after scaling
            st.write("#### Correlation Matrix (Engineered Features)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(eng[feature_cols].corr(), annot=True, fmt='.2f',
                        cmap='coolwarm', ax=ax, linewidths=0.5)
            st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ADVANCED EDA  ★ NEW ★
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "📈 Advanced EDA":
    df     = st.session_state['df']
    df_RFM = st.session_state['df_RFM']

    if df is None or df_RFM is None:
        st.warning("Complete **Data Preparation** first.")
    else:
        st.subheader("Advanced Exploratory Data Analysis")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["RFM Distributions", "Correlation", "Cohort Analysis", "Pair Plots"]
        )

        with tab1:
            st.write("#### RFM Score Distributions")
            fig = make_subplots(rows=1, cols=3, subplot_titles=['Recency','Frequency','Monetary'])
            for i, col in enumerate(['Recency','Frequency','Monetary'], 1):
                fig.add_trace(
                    go.Histogram(x=df_RFM[col], nbinsx=50, marker_color='#667eea', name=col),
                    row=1, col=i
                )
            fig.update_layout(template='plotly_white', showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # RFM Percentile Segmentation
            st.write("#### RFM Percentile Scores (1-5)")
            rfm_s = df_RFM.copy()
            rfm_s['R_Score'] = pd.qcut(rfm_s['Recency'],   5, labels=[5,4,3,2,1]).astype(int)
            rfm_s['F_Score'] = pd.qcut(rfm_s['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
            rfm_s['M_Score'] = pd.qcut(rfm_s['Monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
            rfm_s['RFM_Score'] = rfm_s['R_Score'] + rfm_s['F_Score'] + rfm_s['M_Score']
            rfm_s['Segment'] = pd.cut(
                rfm_s['RFM_Score'],
                bins=[2, 5, 8, 11, 15],
                labels=['Hibernating','At Risk','Loyal','Champions']
            )
            seg_counts = rfm_s['Segment'].value_counts().reset_index()
            fig2 = px.pie(seg_counts, names='Segment', values='count',
                          color_discrete_sequence=px.colors.sequential.Viridis,
                          title="RFM Segment Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.write("#### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(df_RFM.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                        ax=ax, linewidths=0.5, square=True)
            st.pyplot(fig)

            st.write("#### Scatter: Recency vs Monetary (size = Frequency)")
            fig3 = px.scatter(
                df_RFM.reset_index(), x='Recency', y='Monetary', size='Frequency',
                color='Frequency', log_y=True,
                color_continuous_scale='Viridis', template='plotly_white',
                hover_data=['Customer_id'] if 'Customer_id' in df_RFM.reset_index().columns else None
            )
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.write("#### Cohort Retention Analysis")
            df_c = df.copy()
            df_c['order_month']   = df_c['day'].dt.to_period('M')
            df_c['cohort_month']  = df_c.groupby('Customer_id')['day'].transform('min').dt.to_period('M')
            df_c['cohort_index']  = (df_c['order_month'] - df_c['cohort_month']).apply(lambda x: x.n)

            cohort_data = df_c.groupby(['cohort_month','cohort_index'])['Customer_id'].nunique().reset_index()
            cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='Customer_id')
            cohort_size  = cohort_pivot.iloc[:, 0]
            retention    = cohort_pivot.divide(cohort_size, axis=0).round(3) * 100

            fig, ax = plt.subplots(figsize=(14, 7))
            sns.heatmap(retention.iloc[:12, :12], annot=True, fmt='.0f',
                        cmap='YlOrRd_r', ax=ax, linewidths=0.3,
                        cbar_kws={'label': 'Retention %'})
            ax.set_title('Monthly Cohort Retention (%)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Months Since First Purchase')
            ax.set_ylabel('Cohort (First Purchase Month)')
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            st.write("#### Pair Plot (RFM features, sampled)")
            sample = df_RFM.sample(min(500, len(df_RFM)), random_state=42).reset_index()
            fig4 = px.scatter_matrix(
                sample, dimensions=['Recency','Frequency','Monetary'],
                color='Frequency', color_continuous_scale='Viridis',
                template='plotly_white', title='RFM Pair Plot'
            )
            fig4.update_traces(marker=dict(size=3, opacity=0.6))
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODELING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "🤖 Modeling & Evaluation":
    df_eng = st.session_state.get('df_RFM_engineered') if st.session_state.get('df_RFM_engineered') is not None else st.session_state.get('df_RFM')

    if df_eng is None:
        st.warning("Complete **Feature Engineering** (or Data Preparation) first.")
    else:
        st.subheader("KMeans — Elbow + Silhouette")

        sse, sil = {}, {}
        X = df_eng.values

        progress = st.progress(0, text="Computing optimal k...")
        for k in range(2, 15):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sse[k] = km.inertia_
            sil[k] = silhouette_score(X, labels, sample_size=min(2000, len(X)))
            progress.progress((k-1)/13, text=f"Testing k={k}…")
        progress.empty()

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Elbow Method (SSE)', 'Silhouette Score'])
        fig.add_trace(go.Scatter(x=list(sse.keys()), y=list(sse.values()),
                                 mode='lines+markers', name='SSE',
                                 line=dict(color='#667eea', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(sil.keys()), y=list(sil.values()),
                                 mode='lines+markers', name='Silhouette',
                                 line=dict(color='#f093fb', width=2)), row=1, col=2)
        fig.update_layout(template='plotly_white', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        best_k = max(sil, key=sil.get)
        st.info(f"📌 Best k by Silhouette Score: **{best_k}** (score = {sil[best_k]:.4f})")

        n_clusters = st.sidebar.slider('Number of clusters (k)', 2, 15, best_k)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        df_sub = df_eng.copy()
        df_sub['Cluster'] = labels

        # Metrics
        sil_val = silhouette_score(X, labels)
        db_val  = davies_bouldin_score(X, labels)
        ch_val  = calinski_harabasz_score(X, labels)

        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score ↑", f"{sil_val:.4f}")
        c2.metric("Davies-Bouldin ↓",   f"{db_val:.4f}")
        c3.metric("Calinski-Harabasz ↑",f"{ch_val:.0f}")

        # Cluster stats
        raw_eng = st.session_state.get('df_RFM_raw_eng', df_eng)
        df_profile = raw_eng.copy()
        df_profile['Cluster'] = labels
        feature_cols = st.session_state.get('feature_cols', ['Recency','Frequency','Monetary'])
        base_cols = [c for c in ['Recency','Frequency','Monetary'] if c in df_profile.columns]

        cluster_stats = df_profile.groupby('Cluster')[base_cols].mean().round(2)
        cluster_stats['Count'] = df_profile.groupby('Cluster').size()
        cluster_stats['Percent'] = (cluster_stats['Count'] / cluster_stats['Count'].sum() * 100).round(2)
        cluster_stats.index = ['Cluster ' + str(i) for i in cluster_stats.index]
        cluster_stats.reset_index(inplace=True)
        cluster_stats.rename(columns={'index': 'Cluster'}, inplace=True)

        st.write("#### Cluster Statistics")
        st.dataframe(cluster_stats, use_container_width=True)

        # Treemap
        colors = ['#667eea','#f093fb','#4facfe','#43e97b','#fa709a',
                  '#fee140','#30cfd0','#a18cd1','#fda085','#f5576c']
        fig_tm, ax_tm = plt.subplots(figsize=(14, 8))
        if 'RecencyMean' not in cluster_stats.columns:
            cluster_stats = cluster_stats.rename(columns={
                'Recency':   'RecencyMean',
                'Frequency': 'FrequencyMean',
                'Monetary':  'MonetaryMean'
            })
        squarify.plot(
            sizes=cluster_stats['Count'],
            label=[
                f"{row['Cluster']}\n{row.get('RecencyMean','-')} days recency\n"
                f"{row.get('FrequencyMean','-')} orders\n${row.get('MonetaryMean','-')}\n"
                f"{row['Count']} customers ({row['Percent']}%)"
                for _, row in cluster_stats.iterrows()
            ],
            color=colors[:n_clusters], alpha=0.85,
            text_kwargs={'fontsize': 10, 'fontweight': 'bold', 'color': 'white'},
            ax=ax_tm
        )
        ax_tm.set_title("Customer Segments — Treemap", fontsize=18, fontweight='bold')
        ax_tm.axis('off')
        st.pyplot(fig_tm)

        # PCA 2D view
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        pca_df = pd.DataFrame(coords, columns=['PC1','PC2'])
        pca_df['Cluster'] = [f"Cluster {l}" for l in labels]
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                             title='PCA 2D Cluster View',
                             template='plotly_white', opacity=0.7,
                             color_discrete_sequence=colors)
        st.plotly_chart(fig_pca, use_container_width=True)

        if st.button("💾 Export KMeans Model"):
            with open('kmeans_model.pkl', 'wb') as f:
                pickle.dump((model, cluster_stats), f)
            st.session_state['best_model']      = model
            st.session_state['best_model_name'] = 'KMeans'
            st.session_state['cluster_stats']   = cluster_stats
            st.session_state['model_exported']  = True
            st.success("Model exported as `kmeans_model.pkl`")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON  ★ NEW ★
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "🏆 Model Comparison":
    df_eng = st.session_state.get('df_RFM_engineered') if st.session_state.get('df_RFM_engineered') is not None else st.session_state.get('df_RFM')

    if df_eng is None:
        st.warning("Complete **Feature Engineering** first.")
    else:
        st.subheader("🏆 Algorithm Comparison")
        st.write("Compare KMeans, Agglomerative, and DBSCAN on your engineered features.")

        X = df_eng.values
        n_clusters = st.sidebar.slider("k for KMeans & Agglomerative", 2, 10, 3)
        eps_val    = st.sidebar.slider("DBSCAN eps", 0.1, 3.0, 0.5, 0.05)
        min_samp   = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)

        if st.button("🚀 Run Comparison"):
            results = []
            models  = {
                'KMeans':          KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
                'Agglomerative':   AgglomerativeClustering(n_clusters=n_clusters),
                'DBSCAN':          DBSCAN(eps=eps_val, min_samples=min_samp)
            }
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, (name, mdl) in zip(axes, models.items()):
                labels = mdl.fit_predict(X)
                n_found = len(set(labels)) - (1 if -1 in labels else 0)
                noise   = (labels == -1).sum()

                if n_found > 1:
                    try:
                        sil = silhouette_score(X, labels)
                        db  = davies_bouldin_score(X, labels)
                        ch  = calinski_harabasz_score(X, labels)
                    except Exception:
                        sil, db, ch = 0, 0, 0
                else:
                    sil, db, ch = 0, 0, 0

                results.append({
                    'Algorithm': name,
                    'Clusters Found': n_found,
                    'Noise Points': noise,
                    'Silhouette ↑': round(sil, 4),
                    'Davies-Bouldin ↓': round(db, 4),
                    'Calinski-Harabasz ↑': round(ch, 0)
                })

                scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                     c=labels, cmap='tab10', s=5, alpha=0.6)
                ax.set_title(f"{name}\nSilhouette={sil:.3f}", fontsize=11, fontweight='bold')
                ax.axis('off')

            plt.tight_layout()
            st.pyplot(fig)

            res_df = pd.DataFrame(results)
            st.write("#### Metrics Summary")
            st.dataframe(res_df.set_index('Algorithm'), use_container_width=True)

            best = res_df.sort_values('Silhouette ↑', ascending=False).iloc[0]['Algorithm']
            st.success(f"🏆 Best algorithm by Silhouette: **{best}**")
            st.session_state['best_model_name'] = best

# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLUSTER PROFILING  ★ NEW ★
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "🎯 Cluster Profiling":
    cluster_stats = st.session_state.get('cluster_stats')
    df_eng = st.session_state.get('df_RFM_raw_eng') if st.session_state.get('df_RFM_raw_eng') is not None else st.session_state.get('df_RFM')

    if cluster_stats is None:
        st.warning("Export a model in **Modeling & Evaluation** first.")
    else:
        st.subheader("🎯 Cluster Profiling & Business Interpretation")

        st.write("#### Cluster Overview")
        st.dataframe(cluster_stats, use_container_width=True)

        # Radar chart per cluster
        st.write("#### Radar Chart — Cluster Profiles")
        base_cols = [c for c in ['RecencyMean','FrequencyMean','MonetaryMean']
                     if c in cluster_stats.columns]

        if len(base_cols) >= 3:
            radar_df = cluster_stats[['Cluster'] + base_cols].copy()
            for c in base_cols:
                col_min, col_max = radar_df[c].min(), radar_df[c].max()
                if col_max != col_min:
                    radar_df[c] = (radar_df[c] - col_min) / (col_max - col_min)
            # Invert Recency so "lower = better" becomes "higher = better" on radar
            if 'RecencyMean' in radar_df.columns:
                radar_df['RecencyMean'] = 1 - radar_df['RecencyMean']

            categories = base_cols
            fig_radar = go.Figure()
            colors_r = ['#667eea','#f093fb','#4facfe','#43e97b','#fa709a']
            for i, row in radar_df.iterrows():
                vals = row[categories].tolist() + [row[categories[0]]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=row['Cluster'],
                    line=dict(color=colors_r[i % len(colors_r)])
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template='plotly_white', height=500,
                title='Normalized Cluster Profiles (Recency inverted: higher = more recent)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Bar chart comparison
        st.write("#### Side-by-Side Cluster Comparison")
        for metric in base_cols:
            fig_bar = px.bar(
                cluster_stats, x='Cluster', y=metric,
                color='Cluster', template='plotly_white',
                title=metric.replace('Mean', ' (Mean)'),
                color_discrete_sequence=['#667eea','#f093fb','#4facfe','#43e97b','#fa709a']
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Business labels
        st.write("#### 💡 Suggested Business Labels")
        st.markdown("""
        Based on RFM means, assign labels such as:

        | Profile | Recency | Frequency | Monetary | Label |
        |---|---|---|---|---|
        | Low R, High F+M | Recent | Many orders | High spend | 🏆 **Champions** |
        | Mid R, Mid F+M  | Moderate | Regular | Average  | 💛 **Loyal Customers** |
        | High R, Low F   | Long ago | Few orders | Low spend | 😴 **Hibernating** |
        | High R, Mid F+M | Long ago | Some orders | Decent   | ⚠️ **At Risk** |

        Add a `Label` column to the cluster stats and use it in your marketing workflows.
        """)

        csv_download_link(cluster_stats.set_index('Cluster'), 'cluster_profiles.csv',
                          'Download Cluster Profiles CSV')

# ═══════════════════════════════════════════════════════════════════════════════
# 9. PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif choice == "🔮 Predict":
    if not st.session_state.get('model_exported'):
        st.warning("Export a model in **Modeling & Evaluation** first.")
    else:
        with open('kmeans_model.pkl', 'rb') as f:
            model, cluster_stats = pickle.load(f)

        st.subheader("🔮 Predict Customer Segment")
        st.write("#### Cluster Reference")
        st.dataframe(cluster_stats, use_container_width=True)

        tab_single, tab_bulk = st.tabs(["Single Customer", "Bulk CSV Upload"])

        with tab_single:
            st.write("Enter customer transaction details:")
            customer_name = st.text_input('Customer Name / ID:')
            recent_date   = st.date_input(
                'Latest Purchase Date:',
                value=datetime.now().date(),
                min_value=date(1990, 1, 1),
                max_value=date(2026, 12, 31)
            )
            quantity      = st.number_input('Total Quantity Purchased:', min_value=0, value=0)
            monetary      = st.number_input('Total Amount Spent ($):', min_value=0.0, value=0.0)

            if 'df_new' not in st.session_state:
                st.session_state['df_new'] = pd.DataFrame(
                    columns=['Customer_id','day','Quantity','Sales'])

            if st.button("➕ Add Customer"):
                new_row = pd.DataFrame({'Customer_id': [customer_name],
                                        'day': [recent_date],
                                        'Quantity': [quantity], 'Sales': [monetary]})
                st.session_state['df_new'] = pd.concat(
                    [st.session_state['df_new'], new_row], ignore_index=True)

            st.dataframe(st.session_state['df_new'], use_container_width=True)

            if st.button("🔮 Predict"):
                ref_date = st.session_state.get('reference_date', pd.Timestamp.now())
                today = ref_date.date() if isinstance(ref_date, (pd.Timestamp, datetime)) else pd.to_datetime(ref_date).date()
                df_p  = st.session_state['df_new'].copy()
                df_p['day'] = pd.to_datetime(df_p['day'])
                rfm_p = df_p.groupby('Customer_id').agg(
                    Recency   = ('day', lambda x: (today - x.max().date()).days),
                    Frequency = ('Customer_id', 'count'),
                    Monetary  = ('Sales', 'sum')
                )
                # Use only features the model was trained on
                feat = st.session_state.get('feature_cols', ['Recency','Frequency','Monetary'])
                feat = [f for f in feat if f in rfm_p.columns]
                pred = model.predict(rfm_p[feat])
                rfm_p['Predicted_Cluster'] = pred
                st.info(f"Recency is computed using dataset reference date: {today}")
                st.write("#### Prediction Results")
                st.dataframe(rfm_p, use_container_width=True)
                csv_download_link(rfm_p, 'predictions.csv', 'Download Predictions CSV')

        with tab_bulk:
            st.write("Upload a CSV with columns: `Customer_id, day (YYYYMMDD), Quantity, Sales`")
            bulk_file = st.file_uploader("Upload bulk CSV", type=['csv', 'txt'], key='bulk')
            if bulk_file and st.button("🚀 Run Bulk Prediction"):
                try:
                    df_bulk = pd.read_csv(bulk_file, header=None,
                                          names=['Customer_id','day','Quantity','Sales'],
                                          sep=r'\s+', encoding='latin-1')
                    df_bulk['day'] = pd.to_datetime(df_bulk['day'].astype(str), format='%Y%m%d')
                    ref_date = st.session_state.get('reference_date', pd.Timestamp.now())
                    today = ref_date.date() if isinstance(ref_date, (pd.Timestamp, datetime)) else pd.to_datetime(ref_date).date()
                    rfm_bulk = df_bulk.groupby('Customer_id').agg(
                        Recency   = ('day', lambda x: (today - x.max().date()).days),
                        Frequency = ('Customer_id', 'count'),
                        Monetary  = ('Sales', 'sum')
                    )
                    feat = st.session_state.get('feature_cols', ['Recency','Frequency','Monetary'])
                    feat = [f for f in feat if f in rfm_bulk.columns]
                    rfm_bulk['Cluster'] = model.predict(rfm_bulk[feat])
                    st.success(f"✅ Predicted {len(rfm_bulk):,} customers")
                    st.dataframe(rfm_bulk.head(20), use_container_width=True)
                    csv_download_link(rfm_bulk, 'bulk_predictions.csv', 'Download All Predictions')
                except Exception as e:
                    st.error(f"Error: {e}")

    # Feedback
    st.markdown("---")
    st.write("### Feedback")
    fb = st.text_area("Comments or feedback:")
    if st.button("Submit Feedback"):
        save_feedback(fb)
        st.success("Feedback recorded!")
    if os.path.isfile(FEEDBACK_PATH):
        st.write("#### Recent Feedback")
        st.dataframe(
            pd.read_csv(FEEDBACK_PATH).sort_values('Time', ascending=False).head(5),
            use_container_width=True
        )