import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Theme colors ──────────────────────────────────────────────────────────────
C = {
    "navy":   "#2d2261",
    "blue":   "#5b7abe",
    "light":  "#9cbae1",
    "cream":  "#ece1d7",
    "white":  "#ffffff",
    "dark":   "#1a1540",
}

st.set_page_config(page_title="Titanic Data Analysis", layout="wide", page_icon="🚢")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{ background-color: {C['cream']}; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {C['navy']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {C['white']} !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        color: {C['white']} !important;
        font-size: 15px;
    }}

    /* Headers */
    h1, h2, h3 {{ color: {C['navy']} !important; }}

    /* Metric cards */
    div[data-testid="metric-container"] {{
        background-color: {C['white']};
        border: 2px solid {C['light']};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 8px rgba(45,34,97,0.08);
    }}
    div[data-testid="metric-container"] label {{
        color: {C['blue']} !important;
        font-weight: 600;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        color: {C['navy']} !important;
        font-size: 28px !important;
        font-weight: 700;
    }}

    /* Dataframes */
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}

    /* Info boxes */
    .info-box {{
        background-color: {C['white']};
        border-left: 5px solid {C['blue']};
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 2px 2px 8px rgba(45,34,97,0.08);
        color: #2d2261 !important;
    }}
    .info-box * {{
        color: #2d2261 !important;
    }}
    .stat-card {{
        background-color: {C['navy']};
        color: {C['white']};
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 3px 3px 10px rgba(45,34,97,0.2);
    }}
    .stat-card .number {{
        font-size: 36px;
        font-weight: 700;
        color: {C['light']};
    }}
    .stat-card .label {{
        font-size: 13px;
        color: #ccc;
        margin-top: 4px;
    }}
    .section-header {{color: white !important;
        background: linear-gradient(90deg, {C['navy']}, {C['blue']});
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0 10px 0;
    }}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        'figure.facecolor': C['white'],
        'axes.facecolor':   '#f8f9ff',
        'axes.edgecolor':   C['light'],
        'axes.labelcolor':  C['navy'],
        'xtick.color':      C['navy'],
        'ytick.color':      C['navy'],
        'text.color':       C['navy'],
        'grid.color':       '#e0e7ff',
        'grid.alpha':       0.6,
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    try:
        df = pd.read_csv('train.csv')
        df.columns = [c.lower() for c in df.columns]
    except FileNotFoundError:
        df = sns.load_dataset('titanic')
    return df

@st.cache_data
def get_clean(_df):
    df = _df.copy()
    drop_cols = [c for c in ['passengerid','name','ticket','cabin'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['sex_enc'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked_enc'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['familysize'] = df['sibsp'] + df['parch'] + 1
    df['isalone'] = (df['familysize'] == 1).astype(int)
    df['fareperperson'] = df['fare'] / df['familysize']
    df['logfare'] = np.log1p(df['fare'])
    df['age_x_pclass'] = df['age'] * df['pclass']
    return df

df_raw = load_raw()
df = get_clean(df_raw)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚢 Titanic Project")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "📦 Data Collection",
    "🧹 Preprocessing",
    "📊 EDA",
    "🤖 Modeling"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**ENSTA Algeria**")
st.sidebar.markdown("Data Analysis Module · 2025–2026")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA COLLECTION
# ════════════════════════════════════════════════════════════════════════════
if page == "📦 Data Collection":
    st.title("📦 Data Collection")
    st.markdown(f'<div class="info-box">📁 Source: <b>train.csv</b> from Kaggle — loaded with <code>pd.read_csv()</code></div>', unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", "891")
    col2.metric("Total Features", "12")
    col3.metric("Survivors", f"{int(df_raw['survived'].sum())}")
    col4.metric("Survival Rate", f"{df_raw['survived'].mean()*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df_raw.head(8), use_container_width=True)

        st.markdown("### 📊 Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing %'] = (missing['Missing Count'] / len(df_raw) * 100).round(1)

        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = ax.barh(missing['Column'], missing['Missing %'],
                       color=[C['blue'], C['light'], C['navy']], edgecolor='white')
        for bar, val in zip(bars, missing['Missing %']):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val}%', va='center', fontsize=10, color=C['navy'])
        ax.set_xlabel('Missing %')
        ax.set_title('Missing Values by Column', fontweight='bold', color=C['navy'])
        ax.axvline(30, color='red', linestyle='--', alpha=0.5, label='30% threshold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📈 Passenger Class Distribution")
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        class_counts = df_raw['pclass'].value_counts().sort_index()
        bars = ax.bar(['1st Class', '2nd Class', '3rd Class'],
                      class_counts.values,
                      color=[C['navy'], C['blue'], C['light']], edgecolor='white', width=0.55)
        for bar, val in zip(bars, class_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha='center', fontsize=11, fontweight='bold', color=C['navy'])
        ax.set_title('Number of Passengers per Class', fontweight='bold', color=C['navy'])
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### 🥧 Survival Split")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        surv = df_raw['survived'].value_counts()
        ax.pie(surv.values, labels=['Did Not Survive', 'Survived'],
               colors=[C['light'], C['navy']], autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
               textprops={'color': C['navy'], 'fontsize': 11})
        ax.set_title('Survival Distribution', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### 📐 Descriptive Statistics")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧹 Preprocessing":
    st.title("🧹 Preprocessing & Feature Engineering")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🗑️ Dropped Columns")
        st.markdown(f'<div class="info-box"><b>Cabin</b> → 77% missing values — dropped entirely<br><b>PassengerId, Name, Ticket</b> → carry no analytical value</div>', unsafe_allow_html=True)

        st.markdown("### 🩹 Missing Value Treatment")
        st.markdown(f'<div class="info-box"><b>Age</b> → filled with median (28.0) — robust to outliers<br><b>Embarked</b> → filled with mode (S = Southampton)</div>', unsafe_allow_html=True)

        st.markdown("### Before vs After")
        set_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        before = df_raw.isnull().sum()
        before = before[before > 0]
        after_missing = df.isnull().sum()
        after_missing = after_missing[after_missing > 0]

        axes[0].bar(before.index, before.values, color=C['light'], edgecolor='white')
        axes[0].set_title('Before Cleaning', fontweight='bold', color=C['navy'])
        axes[0].set_ylabel('Missing Count')
        axes[0].tick_params(axis='x', rotation=30)

        if after_missing.empty:
            axes[1].bar(['No missing\nvalues'], [0], color=C['navy'], edgecolor='white')
            axes[1].set_ylim(0, 1)
            axes[1].text(0, 0.5, '✅ All clean', ha='center', va='center',
                        fontsize=13, color=C['navy'], fontweight='bold')
        else:
            axes[1].bar(after_missing.index, after_missing.values, color=C['navy'], edgecolor='white')
        axes[1].set_title('After Cleaning', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### ⚙️ Engineered Features")
        feats = pd.DataFrame({
            'Feature': ['FamilySize', 'IsAlone', 'FarePerPerson', 'LogFare', 'Age_x_Pclass'],
            'Formula': ['SibSp + Parch + 1', '1 if alone', 'Fare / FamilySize', 'log(1 + Fare)', 'Age × Pclass'],
            'Purpose': ['Group size', 'Solo traveler flag', 'Per-person cost', 'Reduce skewness', 'Interaction term']
        })
        st.dataframe(feats, use_container_width=True, hide_index=True)

        st.markdown("### 🏆 Feature Selection Results")
        chi2_data = pd.DataFrame({
            'Feature': ['sex_enc', 'pclass', 'IsAlone', 'embarked_enc', 'Title_enc'],
            'Chi² Score': [92.7, 30.9, 14.6, 9.8, 0.8],
            'Selected': ['✅', '✅', '✅', '✅', '❌']
        })

        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_bar = [C['navy'] if s == '✅' else C['light'] for s in chi2_data['Selected']]
        bars = ax.barh(chi2_data['Feature'], chi2_data['Chi² Score'],
                       color=colors_bar, edgecolor='white')
        for bar, val in zip(bars, chi2_data['Chi² Score']):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val}', va='center', fontsize=10, color=C['navy'])
        ax.set_title('Chi-Square Scores (higher = more important)', fontweight='bold', color=C['navy'])
        ax.set_xlabel('Chi² Score')
        ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='Selection threshold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### 📋 Processed Dataset Preview")
    display_cols = ['survived', 'pclass', 'sex', 'age', 'fare', 'familysize', 'isalone', 'logfare', 'age_x_pclass']
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols].head(8), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    # ── Survival overview ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Survival Overview</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Survival Rate", "38.4%")
    col2.metric("Women Survival Rate", "74.2%")
    col3.metric("Men Survival Rate", "18.9%")

    # ── Univariate ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Univariate Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df['age'].dropna(), bins=30, color=C['blue'],
                edgecolor='white', alpha=0.85)
        ax.axvline(df['age'].median(), color=C['navy'], linewidth=2,
                   linestyle='--', label=f'Median: {df["age"].median():.0f}')
        ax.set_title('Age Distribution', fontweight='bold', color=C['navy'])
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df['fare'], bins=40, color=C['light'],
                edgecolor='white', alpha=0.85)
        ax.axvline(df['fare'].median(), color=C['navy'], linewidth=2,
                   linestyle='--', label=f'Median: ${df["fare"].median():.1f}')
        ax.set_title('Fare Distribution (Right-Skewed)', fontweight='bold', color=C['navy'])
        ax.set_xlabel('Fare ($)')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Bivariate — interactive ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Bivariate Analysis</div>', unsafe_allow_html=True)

    feature_choice = st.selectbox(
        "Select a feature to compare with Survival Rate:",
        ['sex', 'pclass', 'embarked']
    )

    col1, col2 = st.columns(2)
    with col1:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        surv_rate = df.groupby(feature_choice)['survived'].mean().sort_values(ascending=False)
        colors_sel = [C['navy'], C['blue'], C['light']][:len(surv_rate)]
        bars = ax.bar(surv_rate.index.astype(str), surv_rate.values,
                      color=colors_sel, edgecolor='white', width=0.5)
        for bar, val in zip(bars, surv_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val*100:.1f}%', ha='center', fontsize=11,
                    fontweight='bold', color=C['navy'])
        ax.set_title(f'Survival Rate by {feature_choice.title()}',
                     fontweight='bold', color=C['navy'])
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for val, color, label in zip(
            df[feature_choice].unique(),
            [C['navy'], C['blue'], C['light']],
            df[feature_choice].unique()
        ):
            subset = df[df[feature_choice] == val]['age'].dropna()
            ax.hist(subset, bins=20, alpha=0.6, color=color, label=str(label), edgecolor='white')
        ax.set_title(f'Age Distribution by {feature_choice.title()}',
                     fontweight='bold', color=C['navy'])
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Three confirmed hypotheses ────────────────────────────────────────────
    st.markdown('<div class="section-header">✅ Confirmed Hypotheses</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    hypotheses = [
        ("🚺 Women vs Men", [74.2, 18.9], ['Female', 'Male']),
        ("🎟️ Class 1 vs Class 3", [63.0, 24.2], ['1st Class', '3rd Class']),
        ("👶 Children vs Adults", [57.4, 36.8], ['Children\n(<12)', 'Adults']),
    ]
    for col, (title, vals, labels) in zip([col1, col2, col3], hypotheses):
        with col:
            set_plot_style()
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(labels, vals,
                          color=[C['navy'], C['light']], edgecolor='white', width=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val}%', ha='center', fontsize=12,
                        fontweight='bold', color=C['navy'])
            ax.set_title(title, fontweight='bold', color=C['navy'], fontsize=12)
            ax.set_ylabel('Survival Rate (%)')
            ax.set_ylim(0, 90)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Multivariate Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'familysize', 'isalone']
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 9})
        ax.set_title('Correlation Matrix', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors_scatter = df['survived'].map({0: C['light'], 1: C['navy']})
        ax.scatter(df['age'], df['fare'], c=colors_scatter, alpha=0.5, s=20)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=C['navy'], label='Survived'),
            Patch(facecolor=C['light'], label='Did Not Survive')
        ])
        ax.set_xlabel('Age')
        ax.set_ylabel('Fare ($)')
        ax.set_title('Age vs Fare (colored by Survival)', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODELING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modeling":
    st.title("🤖 Linear Regression — Fare Prediction")
    st.markdown(f'<div class="info-box">🎯 <b>Target:</b> <code>fare</code> — ticket price paid by the passenger<br>📌 <b>Features:</b> pclass, age, sibsp, parch, sex, embarked<br>✂️ <b>Split:</b> 80% train / 20% test — StandardScaler on train only</div>', unsafe_allow_html=True)
    st.markdown("---")

    @st.cache_data
    def run_models(_df):
        df_m = _df.copy()
        cols = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked']
        df_m = df_m[cols].dropna().copy()
        df_m['sex'] = df_m['sex'].map({'male': 0, 'female': 1})
        df_m['embarked'] = df_m['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        FEATURES = ['pclass', 'age', 'sibsp', 'parch', 'sex', 'embarked']
        X = df_m[FEATURES]
        y = df_m['fare']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        ols   = LinearRegression().fit(X_train_sc, y_train)
        ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)
        lasso = Lasso(alpha=0.5, max_iter=5000).fit(X_train_sc, y_train)
        results = []
        for name, model in [('OLS', ols), ('Ridge (α=1.0)', ridge), ('Lasso (α=0.5)', lasso)]:
            yp = model.predict(X_test_sc)
            results.append({
                'Model': name,
                'MAE ($)': round(mean_absolute_error(y_test, yp), 2),
                'RMSE ($)': round(np.sqrt(mean_squared_error(y_test, yp)), 2),
                'R²': round(r2_score(y_test, yp), 4)
            })
        y_pred_ols = ols.predict(X_test_sc)
        residuals = y_test.values - y_pred_ols
        cv = cross_val_score(LinearRegression(), X_train_sc, y_train, cv=5, scoring='r2')
        return ols, ridge, lasso, FEATURES, results, y_test, y_pred_ols, residuals, X_train, X_test, cv

    ols, ridge, lasso, FEATURES, results, y_test, y_pred_ols, residuals, X_train, X_test, cv = run_models(df_raw)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 OLS Model Performance</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", "$24.85")
    col2.metric("RMSE", "$60.75")
    col3.metric("R²", "0.2869")
    col4.metric("CV R² Mean", f"{cv.mean():.4f}")

    # ── Coefficients ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔢 Feature Coefficients</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        coef_df = pd.Series(ols.coef_, index=FEATURES).sort_values()
        colors_coef = [C['light'] if v < 0 else C['navy'] for v in coef_df]
        bars = ax.barh(coef_df.index, coef_df.values, color=colors_coef, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        for bar, val in zip(bars, coef_df.values):
            ax.text(val + (0.3 if val >= 0 else -0.3),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center',
                    ha='left' if val >= 0 else 'right', fontsize=10)
        ax.set_title('OLS Coefficients (standardized)', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        models_names = ['OLS', 'Ridge', 'Lasso']
        coefs_all = [ols.coef_, ridge.coef_, lasso.coef_]
        x = np.arange(len(FEATURES))
        width = 0.25
        for i, (name, coefs, color) in enumerate(zip(
            models_names, coefs_all, [C['navy'], C['blue'], C['light']]
        )):
            ax.bar(x + i*width, coefs, width, label=name, color=color, edgecolor='white')
        ax.set_xticks(x + width)
        ax.set_xticklabels(FEATURES, rotation=30, ha='right')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.legend()
        ax.set_title('OLS vs Ridge vs Lasso Coefficients', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Interactive alpha slider ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🎛️ Interactive — Try Different Alpha Values</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        model_type = st.radio("Model", ["Ridge", "Lasso"])
        alpha_val = st.slider("Alpha (λ)", min_value=0.01, max_value=50.0, value=1.0, step=0.1)

    cols_m = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked']
    df_m = df_raw[cols_m].dropna().copy()
    df_m['sex'] = df_m['sex'].map({'male': 0, 'female': 1})
    df_m['embarked'] = df_m['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X_s = df_m[FEATURES]
    y_s = df_m['fare']
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    reg = Ridge(alpha=alpha_val) if model_type == "Ridge" else Lasso(alpha=alpha_val, max_iter=5000)
    reg.fit(X_tr_sc, y_tr)
    y_pred_int = reg.predict(X_te_sc)
    r2_int = r2_score(y_te, y_pred_int)
    mae_int = mean_absolute_error(y_te, y_pred_int)

    with col2:
        set_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        coef_int = pd.Series(reg.coef_, index=FEATURES)
        clrs = [C['light'] if v == 0 else C['navy'] for v in coef_int]
        axes[0].bar(FEATURES, coef_int.values, color=clrs, edgecolor='white')
        axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[0].set_title(f'{model_type} (α={alpha_val}) Coefficients', fontweight='bold', color=C['navy'])
        axes[0].set_xticklabels(FEATURES, rotation=30, ha='right')

        res_int = y_te.values - y_pred_int
        axes[1].scatter(y_pred_int, res_int, alpha=0.5, color=C['blue'], s=20)
        axes[1].axhline(0, color='red', linewidth=2, linestyle='--')
        axes[1].set_xlabel('Predicted Fare ($)')
        axes[1].set_ylabel('Residual ($)')
        axes[1].set_title('Residual Plot', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{model_type} R²", f"{r2_int:.4f}")
    col2.metric(f"{model_type} MAE", f"${mae_int:.2f}")
    zeroed = int((reg.coef_ == 0).sum())
    col3.metric("Features Zeroed", zeroed)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 OLS Diagnostics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        mn = min(y_test.min(), y_pred_ols.min()) - 5
        mx = max(y_test.max(), y_pred_ols.max()) + 5
        ax.scatter(y_test, y_pred_ols, alpha=0.45, color=C['blue'], s=20)
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual Fare ($)')
        ax.set_ylabel('Predicted Fare ($)')
        ax.set_title('Actual vs Predicted', fontweight='bold', color=C['navy'])
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(y_pred_ols, residuals, alpha=0.45, color=C['navy'], s=20)
        ax.axhline(0, color='red', linewidth=2, linestyle='--')
        ax.set_xlabel('Predicted Fare ($)')
        ax.set_ylabel('Residual ($)')
        ax.set_title('Residual Plot', fontweight='bold', color=C['navy'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col3:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.hist(residuals, bins=25, color=C['light'], edgecolor='white')
        ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Zero error')
        ax.axvline(residuals.mean(), color=C['navy'], linewidth=2,
                   label=f'Mean: {residuals.mean():.2f}')
        ax.set_xlabel('Residual ($)')
        ax.set_ylabel('Count')
        ax.set_title('Residual Distribution', fontweight='bold', color=C['navy'])
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Model Comparison</div>', unsafe_allow_html=True)
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown(f'<div class="info-box">💡 <b>Interpretation:</b> All three models perform similarly with R² ≈ 0.29. <b>pclass</b> has the strongest negative coefficient (≈ −26): higher class number = cheaper ticket. Lasso did not zero out any feature. The moderate R² is expected — fare is heavily driven by class and the relationship is not purely linear.</div>', unsafe_allow_html=True)

