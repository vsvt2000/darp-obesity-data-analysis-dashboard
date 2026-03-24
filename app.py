# ================================================================
#  DataLab Studio — Obesity Level Estimation
#  Run with: streamlit run app.py
#  Place your dataset CSV in the same folder as this file
# ================================================================

import scipy
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obesity Level Estimator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.top-banner {
    background: linear-gradient(135deg, #0F6E56 0%, #185FA5 100%);
    padding: 24px 32px; border-radius: 14px;
    color: white; margin-bottom: 24px;
}
.banner-title { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.banner-sub   { font-size: 14px; opacity: 0.85; }

.metric-card {
    padding: 16px 18px; border-radius: 12px; margin-bottom: 8px;
}
.metric-label { font-size: 12px; color: #888780; margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 700; }

.section-card {
    background: #ffffff; border: 1px solid #E8E6DF;
    border-radius: 14px; padding: 20px 24px; margin-bottom: 16px;
}
.section-title {
    font-size: 16px; font-weight: 600; color: #2C2C2A;
    border-left: 4px solid #1D9E75; padding-left: 10px;
    margin-bottom: 14px;
}
.result-box {
    background: #E1F5EE; border-left: 3px solid #1D9E75;
    padding: 10px 14px; border-radius: 8px;
    font-family: monospace; font-size: 13px; color: #0F6E56;
    margin-top: 8px;
}
.chat-ai {
    background: #E1F5EE; color: #2C2C2A;
    padding: 10px 14px; border-radius: 12px 12px 12px 2px;
    font-size: 13px; margin-bottom: 8px;
}
.chat-user {
    background: #E6F1FB; color: #2C2C2A;
    padding: 10px 14px; border-radius: 12px 12px 2px 12px;
    font-size: 13px; margin-bottom: 8px; text-align: right;
}
</style>
""", unsafe_allow_html=True)

# ── Gemini setup ─────────────────────────────────────────────────────────────
# 🔑 Replace with your Gemini API key from https://aistudio.google.com
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ── Session state ─────────────────────────────────────────────────────────────
if "df"           not in st.session_state: st.session_state.df           = None
if "chat_history" not in st.session_state: st.session_state.chat_history = [
    {"role": "ai", "text": "Hi! I'm your AI assistant for this obesity analysis project. Load the dataset and explore the sections — then ask me anything!"}
]
if "results"      not in st.session_state: st.session_state.results      = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 DataLab Studio")
    st.caption("Obesity Level Estimation")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload your dataset CSV",
        type=["csv"],
        help="Upload ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw.dropna(inplace=True)
        df_raw['TUE']  = pd.Series(list(round(df_raw['TUE'])))
        df_raw['FAF']  = pd.Series(list(round(df_raw['FAF'])))
        df_raw['CH2O'] = pd.Series(list(round(df_raw['CH2O'])))
        df_raw['NCP']  = pd.Series(list(round(df_raw['NCP'])))
        df_raw['FCVC'] = pd.Series(list(round(df_raw['FCVC'])))

        # Age categories used across EDA
        df_raw['AGE_Category'] = pd.cut(
            df_raw['Age'], bins=[0, 20, 40, 60],
            labels=['Youth', 'Early Middle Age', 'Later Middle Age']
        )
        st.session_state.df = df_raw
        st.success(f"✅ Loaded {len(df_raw)} rows")

    st.markdown("---")
    section = st.radio("Navigate", [
        "📊 Overview",
        "🔍 EDA",
        "🧪 Hypothesis Testing",
        "🧩 EFA",
        "🔵 Clustering",
        "📈 Linear Regression",
        "🎯 Classification",
        "🤖 AI Assistant",
    ])

# ── Top banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <div class="banner-title">🧬 Obesity Level Estimation</div>
  <div class="banner-sub">Based on Eating Habits & Physical Condition — DARP Project</div>
</div>
""", unsafe_allow_html=True)

df = st.session_state.df

# ── Helper: save figure as streamlit plot ────────────────────────────────────
def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if section == "📊 Overview":
    if df is None:
        st.info("👈 Please upload your dataset CSV using the sidebar to get started.")
    else:
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card" style="background:#E1F5EE"><div class="metric-label">Total Records</div><div class="metric-value" style="color:#0F6E56">{len(df):,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card" style="background:#E6F1FB"><div class="metric-label">Features</div><div class="metric-value" style="color:#185FA5">{df.shape[1]}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card" style="background:#FAEEDA"><div class="metric-label">Obesity Classes</div><div class="metric-value" style="color:#854F0B">{df["NObeyesdad"].nunique()}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card" style="background:#FBEAF0"><div class="metric-label">Missing Values</div><div class="metric-value" style="color:#993556">{df.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Dataset Preview**")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("**Column Summary**")
        st.dataframe(df.describe().transpose(), use_container_width=True)

        st.session_state.results["overview"] = f"Dataset: {len(df)} rows, {df.shape[1]} cols, {df['NObeyesdad'].nunique()} obesity classes, 0 missing values."

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif section == "🔍 EDA":
    if df is None:
        st.info("👈 Please upload your dataset CSV first.")
    else:
        st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        eda_var = st.selectbox("Select variable to explore", [
            "Obesity Level (NObeyesdad)",
            "Transportation (MTRANS)",
            "Alcohol Consumption (CALC)",
            "Technology Use (TUE)",
            "Physical Activity (FAF)",
            "Calorie Monitoring (SCC)",
            "Water Consumption (CH2O)",
            "Smoking (SMOKE)",
            "Food Between Meals (CAEC)",
            "Main Meals (NCP)",
            "Vegetable Consumption (FCVC)",
            "High-Caloric Food (FAVC)",
            "Family History Overweight",
            "Weight",
            "Height",
            "Age",
            "Gender",
        ])

        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            # ── Obesity Level ──────────────────────────────────────
            if eda_var == "Obesity Level (NObeyesdad)":
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x='NObeyesdad', data=df,
                              order=df['NObeyesdad'].value_counts().index,
                              palette='viridis', ax=ax)
                ax.set_title("Distribution of Obesity Levels")
                ax.set_xlabel("Obesity Level"); ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                show_plot(fig)

            elif eda_var == "Transportation (MTRANS)":
                mc = df['MTRANS'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(mc, labels=mc.index, autopct='%1.1f%%', startangle=140)
                ax.set_title("Distribution of Transportation Modes")
                show_plot(fig)

            elif eda_var == "Alcohol Consumption (CALC)":
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.countplot(y='CALC', data=df,
                              order=df['CALC'].value_counts().index,
                              palette='coolwarm', ax=ax)
                ax.set_title("Alcohol Consumption Distribution")
                show_plot(fig)

            elif eda_var == "Technology Use (TUE)":
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x='AGE_Category', y='TUE', data=df,
                            order=['Youth', 'Early Middle Age', 'Later Middle Age'],
                            palette='magma', ax=ax)
                ax.set_title("Technology Use by Age Group")
                show_plot(fig)

            elif eda_var == "Physical Activity (FAF)":
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df['FAF'], bins=10, ax=ax)
                ax.set_title("Physical Activity Frequency Distribution")
                show_plot(fig)

            elif eda_var == "Calorie Monitoring (SCC)":
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.countplot(x='SCC', data=df, palette='pastel', ax=ax)
                ax.set_title("Calories Consumption Monitoring (SCC)")
                show_plot(fig)

            elif eda_var == "Water Consumption (CH2O)":
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df['CH2O'], ax=ax)
                ax.set_title("Daily Water Consumption (CH2O)")
                show_plot(fig)

            elif eda_var == "Smoking (SMOKE)":
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.countplot(x='SMOKE', data=df, palette='Set1', ax=ax)
                ax.set_title("Smoking Habit Distribution")
                show_plot(fig)

            elif eda_var == "Food Between Meals (CAEC)":
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.countplot(x='CAEC', data=df,
                              order=df['CAEC'].value_counts().index,
                              palette='cubehelix', ax=ax)
                ax.set_title("Consumption of Food Between Meals (CAEC)")
                show_plot(fig)

            elif eda_var == "Main Meals (NCP)":
                df['NCP_category'] = pd.cut(df['NCP'], bins=[0, 2, 3, 5],
                    labels=['Low (1-2)', 'Medium (3)', 'High (4+)'])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='NCP_category', data=df,
                              order=['Low (1-2)', 'Medium (3)', 'High (4+)'],
                              palette='viridis', ax=ax)
                ax.set_title("Number of Main Meals (Categorized)")
                show_plot(fig)

            elif eda_var == "Vegetable Consumption (FCVC)":
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df['FCVC'], ax=ax)
                ax.set_title("Frequency of Vegetable Consumption (FCVC)")
                show_plot(fig)

            elif eda_var == "High-Caloric Food (FAVC)":
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.countplot(x='FAVC', data=df, palette='cool', ax=ax)
                ax.set_title("Frequent Consumption of High-Caloric Food")
                show_plot(fig)

            elif eda_var == "Family History Overweight":
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.countplot(x='family_history_with_overweight', data=df,
                              palette='Set2', ax=ax)
                ax.set_title("Family History with Overweight")
                show_plot(fig)

            elif eda_var == "Weight":
                df['Weight_category'] = pd.cut(df['Weight'],
                    bins=[40, 60, 80, 100, 120, 200],
                    labels=['40-60', '60-80', '80-100', '100-120', '120+'])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Weight_category', data=df, palette='viridis', ax=ax)
                ax.set_title("Weight Distribution (Categorized)")
                show_plot(fig)

            elif eda_var == "Height":
                df['Height_category'] = pd.cut(df['Height'],
                    bins=[1.4, 1.55, 1.7, 1.85, 2.1],
                    labels=['Short', 'Average', 'Tall', 'Very Tall'])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Height_category', data=df, palette='coolwarm', ax=ax)
                ax.set_title("Height Distribution (Categorized)")
                show_plot(fig)

            elif eda_var == "Age":
                df['Age_category'] = pd.cut(df['Age'],
                    bins=[10, 20, 30, 40, 50, 60],
                    labels=['10-20', '20-30', '30-40', '40-50', '50-60'])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Age_category', data=df, palette='viridis', ax=ax)
                ax.set_title("Age Distribution (Categorized)")
                show_plot(fig)

            elif eda_var == "Gender":
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.countplot(x='Gender', data=df, palette='Set3', ax=ax)
                ax.set_title("Gender Distribution")
                show_plot(fig)

        with col_right:
            # Variable name extraction
            col_map = {
                "Obesity Level (NObeyesdad)": "NObeyesdad",
                "Transportation (MTRANS)": "MTRANS",
                "Alcohol Consumption (CALC)": "CALC",
                "Technology Use (TUE)": "TUE",
                "Physical Activity (FAF)": "FAF",
                "Calorie Monitoring (SCC)": "SCC",
                "Water Consumption (CH2O)": "CH2O",
                "Smoking (SMOKE)": "SMOKE",
                "Food Between Meals (CAEC)": "CAEC",
                "Main Meals (NCP)": "NCP",
                "Vegetable Consumption (FCVC)": "FCVC",
                "High-Caloric Food (FAVC)": "FAVC",
                "Family History Overweight": "family_history_with_overweight",
                "Weight": "Weight",
                "Height": "Height",
                "Age": "Age",
                "Gender": "Gender",
            }
            col = col_map[eda_var]
            st.markdown(f"**{eda_var} — Stats**")
            st.dataframe(pd.DataFrame(df[col].describe()).T, use_container_width=True)
            st.markdown(f"**Value Counts**")
            st.dataframe(pd.DataFrame(df[col].value_counts()).head(10), use_container_width=True)

        st.session_state.results["eda"] = f"EDA explored: {eda_var}. Dataset has {len(df)} records with variables covering diet, physical activity, weight, height, age, and gender."

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — HYPOTHESIS TESTING
# ════════════════════════════════════════════════════════════════════════════
elif section == "🧪 Hypothesis Testing":
    if df is None:
        st.info("👈 Please upload your dataset CSV first.")
    else:
        st.markdown('<div class="section-title">Hypothesis Testing</div>', unsafe_allow_html=True)

        # T-test
        st.markdown("#### H1 — Gender vs Weight (Independent T-Test)")
        male_weight   = df[df['Gender'] == 'Male']['Weight']
        female_weight = df[df['Gender'] == 'Female']['Weight']
        t_stat, p_val = stats.ttest_ind(male_weight, female_weight)
        conclusion_t  = "Reject H₀: Significant difference in weight between genders." if p_val < 0.05 else "Fail to Reject H₀: No significant difference."
        col1, col2 = st.columns(2)
        with col1:
            st.metric("T-Statistic", f"{t_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_val:.6f}")
        st.markdown(f'<div class="result-box">{"✅" if p_val < 0.05 else "❌"} {conclusion_t}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Chi-square
        st.markdown("#### H2 — High-Caloric Food (FAVC) vs Obesity Level (Chi-Square)")
        table = pd.crosstab(df['FAVC'], df['NObeyesdad'])
        chi2, p_chi, dof, _ = stats.chi2_contingency(table)
        conclusion_chi = "Reject H₀: FAVC and obesity level are related." if p_chi < 0.05 else "Fail to Reject H₀: No relationship found."
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Chi-Square", f"{chi2:.4f}")
        with col4:
            st.metric("P-Value", f"{p_chi:.6f}")
        st.markdown(f'<div class="result-box">{"✅" if p_chi < 0.05 else "❌"} {conclusion_chi}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Pearson
        st.markdown("#### H3 — Physical Activity (FAF) vs Weight (Pearson Correlation)")
        corr, p_corr = stats.pearsonr(df['FAF'], df['Weight'])
        conclusion_corr = "Reject H₀: Significant correlation between FAF and weight." if p_corr < 0.05 else "Fail to Reject H₀: No significant correlation."
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Correlation Coefficient", f"{corr:.4f}")
        with col6:
            st.metric("P-Value", f"{p_corr:.6f}")
        st.markdown(f'<div class="result-box">{"✅" if p_corr < 0.05 else "❌"} {conclusion_corr}</div>', unsafe_allow_html=True)

        st.session_state.results["hypothesis"] = (
            f"T-test (Gender vs Weight): t={t_stat:.4f}, p={p_val:.6f} → {conclusion_t} | "
            f"Chi-square (FAVC vs Obesity): χ²={chi2:.4f}, p={p_chi:.6f} → {conclusion_chi} | "
            f"Pearson (FAF vs Weight): r={corr:.4f}, p={p_corr:.6f} → {conclusion_corr}"
        )

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — EFA
# ════════════════════════════════════════════════════════════════════════════
elif section == "🧩 EFA":
    if df is None:
        st.info("👈 Please upload your dataset CSV first.")
    else:
        st.markdown('<div class="section-title">Exploratory Factor Analysis</div>', unsafe_allow_html=True)
        try:
            from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
            from factor_analyzer import FactorAnalyzer

            if not hasattr(scipy, 'sum'):
                scipy.sum = np.sum
            

            df_efa = df[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']].copy()
            le = LabelEncoder()
            for col in df_efa.select_dtypes(include='int').columns:
                df_efa[col] = le.fit_transform(df_efa[col])

            chi_sq, p_bart = calculate_bartlett_sphericity(df_efa)
            kmo_all, kmo_model = calculate_kmo(df_efa)

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Bartlett χ²", f"{chi_sq:.2f}")
            with col2: st.metric("Bartlett p-value", f"{p_bart:.6f}")
            with col3: st.metric("KMO Score", f"{kmo_model:.4f}")

            suitable = p_bart < 0.05 and kmo_model > 0.6
            st.markdown(
                f'<div class="result-box">{"✅ Data is suitable for EFA." if suitable else "❌ Data may not be suitable for EFA."}</div>',
                unsafe_allow_html=True
            )

            # Scree plot
            fa = FactorAnalyzer(n_factors=8, rotation=None)
            fa.fit(df_efa)
            ev, _ = fa.get_eigenvalues()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(range(1, len(ev) + 1), ev, marker='o', color='#1D9E75')
            ax.axhline(y=1, color='#E24B4A', linestyle='--', label='Eigenvalue = 1')
            ax.set_title("Scree Plot")
            ax.set_xlabel("Factor Number")
            ax.set_ylabel("Eigenvalue")
            ax.legend()
            show_plot(fig)

            st.session_state.results["efa"] = f"EFA — Bartlett p={p_bart:.6f}, KMO={kmo_model:.4f}. {'Suitable for EFA.' if suitable else 'Not suitable for EFA.'}"
            fa_full = FactorAnalyzer(n_factors=8, rotation=None)
            fa_full.fit(df_efa)
            ev, _ = fa_full.get_eigenvalues()

            # Factors with eigenvalue > 1 (Kaiser criterion)
            n_factors_kaiser = int(sum(ev > 1))
            st.markdown(f"#### Eigenvalues (Kaiser Criterion: eigenvalue > 1)")
            st.markdown(f"**{n_factors_kaiser} factors recommended** based on Kaiser criterion.")

            # ── Factor Loadings ───────────────────────────────────
            st.markdown(f"#### Factor Loadings (Top {n_factors_kaiser} Factors, Varimax Rotation)")
            fa_final = FactorAnalyzer(n_factors=n_factors_kaiser, rotation='varimax')
            fa_final.fit(df_efa)

            loadings = pd.DataFrame(
                fa_final.loadings_,
                index=df_efa.columns,
                columns=[f"Factor {i+1}" for i in range(n_factors_kaiser)]
            ).round(4)

            # Highlight strong loadings (|loading| > 0.4)
            st.markdown("Values **> 0.4** or **< -0.4** indicate strong factor relationships.")
            st.dataframe(
                loadings.style.background_gradient(cmap='RdYlGn', axis=None, vmin=-1, vmax=1),
                use_container_width=True
            )

            # ── Most relevant variables per factor ────────────────
            st.markdown("---")
            st.markdown("#### Most Relevant Variables per Factor")
            st.caption("Variables with strongest loading (|value| > 0.4) for each factor")

            for factor in loadings.columns:
                strong = loadings[factor][loadings[factor].abs() > 0.4].sort_values(
                    key=abs, ascending=False)
                if len(strong) > 0:
                    st.markdown(f"**{factor}**")
                    for var, val in strong.items():
                        direction = "↑ positive" if val > 0 else "↓ negative"
                        st.markdown(
                            f'<div class="result-box">{var} &nbsp;|&nbsp; loading: {val:.4f} &nbsp;|&nbsp; {direction}</div>',
                            unsafe_allow_html=True)
                    st.markdown("")

            # ── Variance explained ────────────────────────────────
            st.markdown("---")
            st.markdown("#### Variance Explained")
            variance = fa_final.get_factor_variance()
            var_df = pd.DataFrame(
                variance,
                index=['SS Loadings', 'Proportion Variance', 'Cumulative Variance'],
                columns=[f"Factor {i+1}" for i in range(n_factors_kaiser)]
            ).round(4)
            st.dataframe(var_df, use_container_width=True)

            total_var = variance[2][-1] * 100
            st.markdown(
                f'<div class="result-box">✅ {n_factors_kaiser} factors explain {total_var:.1f}% of total variance.</div>',
                unsafe_allow_html=True)

            st.session_state.results["efa"] = (
                f"EFA — Bartlett p={p_bart:.6f}, KMO={kmo_model:.4f}. "
                f"{n_factors_kaiser} factors retained (Kaiser criterion). "
                f"Total variance explained: {total_var:.1f}%."
            )

        except ImportError:
            st.warning("Run `pip install factor_analyzer` to enable this section.")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
elif section == "🔵 Clustering":
    if df is None:
        st.info("👈 Please upload your dataset CSV first.")
    else:
        st.markdown('<div class="section-title">KMeans Clustering</div>', unsafe_allow_html=True)

        df_cluster = df[['Gender', 'Age', 'Height', 'Weight',
                         'family_history_with_overweight', 'FAVC', 'FCVC',
                         'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
                         'TUE', 'CALC', 'MTRANS', 'NObeyesdad']].copy()
        le = LabelEncoder()
        for col in df_cluster.select_dtypes(include='object').columns:
            df_cluster[col] = le.fit_transform(df_cluster[col])

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)

        k_range = range(2, 11)
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(list(k_range), inertias, marker='o', color='#1D9E75')
            ax.set_title("Elbow Method — Optimal K")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Inertia")
            show_plot(fig)

        with col2:
            k = st.slider("Select K", min_value=2, max_value=10, value=4)
            km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels   = km_final.fit_predict(X_scaled)
            df_cluster['Cluster'] = labels

            cluster_counts = pd.Series(labels).value_counts().sort_index()
            st.markdown(f"**Cluster Distribution (K={k})**")
            st.dataframe(pd.DataFrame({"Cluster": cluster_counts.index,
                                       "Count":   cluster_counts.values}),
                         use_container_width=True)

        st.session_state.results["clustering"] = f"KMeans clustering with K={k}. Cluster sizes: {dict(cluster_counts)}."



# ════════════════════════════════════════════════════════════════════════════
#  SECTION — LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════════════
elif section == "📈 Linear Regression":
    if df is None:
        st.info("👈 Please upload your dataset CSV first.")
    else:
        st.markdown('<div class="section-title">Linear Regression — Predicting Weight</div>', unsafe_allow_html=True)

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        df_reg = df[['Gender', 'Age', 'Height', 'Weight',
                         'family_history_with_overweight', 'FAVC', 'FCVC',
                         'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
                         'TUE', 'CALC', 'MTRANS', 'NObeyesdad']].copy()
        le = LabelEncoder()
        for col in df_reg.select_dtypes(include='object').columns:
            df_reg[col] = le.fit_transform(df_reg[col])

        X = df_reg.drop('Weight', axis=1)
        y = df_reg['Weight']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mse = mean_squared_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)

        # ── Metrics ──────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.metric("R² Score", f"{r2:.4f}")
        with c2:
            st.metric("Mean Squared Error", f"{mse:.4f}")

        st.markdown(f'<div class="result-box">{"✅ Good fit!" if r2 > 0.7 else "⚠️ Model may need improvement."} R² = {r2:.4f} — model explains {r2*100:.1f}% of variance in Weight.</div>', unsafe_allow_html=True)

        # ── Actual vs Predicted plot ──────────────────────────────
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Actual vs Predicted Weight**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.4, color='#1D9E75', edgecolors='none')
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', linewidth=1.5, label='Perfect fit')
            ax.set_xlabel("Actual Weight")
            ax.set_ylabel("Predicted Weight")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            show_plot(fig)

        with col2:
            st.markdown("**Residuals Distribution**")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(residuals, bins=30, kde=True,
                         color='#185FA5', ax=ax)
            ax.axvline(0, color='red', linestyle='--')
            ax.set_title("Residuals Distribution")
            ax.set_xlabel("Residual (Actual - Predicted)")
            show_plot(fig)

        # ── Feature importance ────────────────────────────────────
        st.markdown("**Top 10 Feature Importances**")
        coef_df = pd.DataFrame({
            'Feature':     X.columns,
            'Coefficient': abs(model.coef_)
        }).sort_values('Coefficient', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x='Coefficient', y='Feature',
                    data=coef_df, palette='viridis', ax=ax)
        ax.set_title("Top 10 Features by Coefficient Magnitude")
        show_plot(fig)

        st.session_state.results["regression"] = (
            f"Linear Regression (predicting Weight) — "
            f"R²: {r2:.4f}, MSE: {mse:.4f}. "
            f"Model explains {r2*100:.1f}% of variance."
        )

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════


elif section == "🎯 Classification":
    if df is None:
        st.info("👈 Please upload your dataset CSV first.")
    else:
        st.markdown('<div class="section-title">Linear Discriminant Analysis — Obesity Classification</div>', unsafe_allow_html=True)

        df_model = df[['Gender', 'Age', 'Height', 'Weight',
                       'family_history_with_overweight', 'FAVC', 'FCVC',
                       'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
                       'TUE', 'CALC', 'MTRANS', 'NObeyesdad']].copy()
        le = LabelEncoder()
        for col in df_model.select_dtypes(include='object').columns:
            df_model[col] = le.fit_transform(df_model[col])

        X = df_model.drop('NObeyesdad', axis=1)
        y = df_model['NObeyesdad']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_s, y_train)
        y_pred   = lda.predict(X_test_s)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, output_dict=True)
        cm       = confusion_matrix(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy * 100:.2f}%")
            st.markdown("**Classification Report**")
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
        with col2:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', ax=ax)
            ax.set_title("Confusion Matrix — LDA")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            show_plot(fig)

        st.session_state.results["classification"] = (
            f"LDA Classification — Accuracy: {accuracy * 100:.2f}%. "
            f"Macro F1: {report['macro avg']['f1-score']:.3f}. "
            f"Classes: {df['NObeyesdad'].nunique()} obesity levels."
        )

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — AI ASSISTANT
# ════════════════════════════════════════════════════════════════════════════
elif section == "🤖 AI Assistant":
    st.markdown('<div class="section-title">AI Assistant</div>', unsafe_allow_html=True)
    st.caption("Ask anything about your obesity analysis project")

    for msg in st.session_state.chat_history:
        if msg["role"] == "ai":
            st.markdown(f'<div class="chat-ai"><b>🤖 AI</b><br>{msg["text"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-user">{msg["text"]}<br><b>You</b></div>',
                        unsafe_allow_html=True)

    question = st.chat_input("Ask about your analysis, results, or methodology…")
    if question:
        st.session_state.chat_history.append({"role": "user", "text": question})

        results_summary = "\n".join([
            f"{k}: {v}" for k, v in st.session_state.results.items()
        ]) or "No analysis run yet."

        dataset_info = (
            f"Dataset: {len(df)} rows, {df.shape[1]} cols" if df is not None
            else "Dataset not loaded yet."
        )

        prompt = f"""You are an expert AI assistant for a data science project on obesity level estimation.

Project: Estimation of Obesity Levels Based on Eating Habits and Physical Condition.
Dataset: {dataset_info}
Variables include: Gender, Age, Height, Weight, eating habits (FAVC, FCVC, NCP, CAEC, CH2O),
lifestyle (SMOKE, SCC, FAF, TUE, CALC, MTRANS), family_history_with_overweight, NObeyesdad (target).

Current analysis results:
{results_summary}

User question: {question}

Answer concisely and helpfully. If referencing statistical results, explain what they mean in plain English."""

        try:
            response = gemini_model.generate_content(prompt)
            reply    = response.text
        except Exception as e:
            reply = f"Gemini error: {e}. Please check your API key in the script."

        st.session_state.chat_history.append({"role": "ai", "text": reply})
        st.rerun()
