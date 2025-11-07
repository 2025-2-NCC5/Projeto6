import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dash import dcc, html
import plotly.express as px
import streamlit as st
import locale
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df_0 = pd.read_excel("https://github.com/2025-2-NCC5/Projeto6/raw/main/documentos/base_cannoli.xlsx")

# Tratar valores nulos
df = df_0.fillna({
    'totalAmount': 0
})

# Padronizar datas
for col in df.columns:
    if any(x in col.lower() for x in ['sendat', 'purchasedat', 'dateofbirth']):
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Criar features temporais
# 1 - Campanha
if 'sendAt' in df.columns:
    df['sendAt_month'] = df['sendAt'].dt.month
    df['sendAt_week'] = df['sendAt'].dt.isocalendar().week
    df['sendAt_weekday_name'] = df['sendAt'].dt.day_name()

    dias_pt = {
        'Monday': 'Segunda-feira',
        'Tuesday': 'Terça-feira',
        'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira',
        'Friday': 'Sexta-feira',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    df['sendAt_weekday_name'] = df['sendAt_weekday_name'].map(dias_pt)

# 2 - Compra
if 'purchasedAt' in df.columns:
    df['purchasedAt_month'] = df['purchasedAt'].dt.month
    df['purchasedAt_week'] = df['purchasedAt'].dt.isocalendar().week
    df['purchasedAt_weekday_name'] = df['purchasedAt'].dt.day_name()

# 3 - Idade
if 'dateOfBirth' in df.columns:
    today = pd.to_datetime('today').normalize()
    df['age'] = np.floor((today - df['dateOfBirth']).dt.days / 365.25)

# 4 - Cliente impactado pela campanha
if {'sendAt', 'purchasedAt'}.issubset(df.columns):
    df['clienteImpactado'] = np.where(
        (df['purchasedAt'] >= df['sendAt']) &
        (df['purchasedAt'] <= df['sendAt'] + pd.Timedelta(days=7)),
        1, 0
    )
else:
    df['clienteImpactado'] = 0

# Codificar apenas colunas específicas
cat_cols = ['gender', 'salesChannel']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Criação de agregações
campanhas = df.groupby('campaignName').agg({
    'response': 'count',
    'totalAmount': 'mean',
    'clienteImpactado': 'mean'
}).reset_index()
campanhas.rename(columns={
    'response': 'clientes_impactados',
    'totalAmount': 'gasto_medio',
    'clienteImpactado': 'taxa_impacto'
}, inplace=True)

idade = df.groupby('age').agg({
    'response': 'count',
    'totalAmount': 'mean',
    'clienteImpactado': 'mean'
}).reset_index()
idade.rename(columns={
    'response': 'clientes_impactados',
    'totalAmount': 'gasto_medio',
    'clienteImpactado': 'taxa_impacto'
}, inplace=True)

if 'gender' in df.columns:
    genero = df.groupby('gender').agg({
        'response': 'count',
        'totalAmount': 'mean',
        'clienteImpactado': 'mean'
    }).reset_index()
    genero.rename(columns={
        'response': 'clientes_impactados',
        'totalAmount': 'gasto_medio',
        'clienteImpactado': 'taxa_impacto'
    }, inplace=True)
else:
    genero = pd.DataFrame(columns=['gender', 'clientes_impactados', 'gasto_medio', 'taxa_impacto'])

df['target_comprou'] = (df['totalAmount'] > 0).astype(int)
df['campaignName'] = df['campaignName'].fillna('Nenhuma')
df['response'] = df['response'].fillna('Nao_Respondeu')
df['salesChannel'] = df['salesChannel'].fillna('Desconhecido')

numeric_features = ['age']
categorical_features = [
    'gender', 'contrycode', 'areacode', 'campaignName',
    'response', 'sendAt_weekday_name'
]
features = numeric_features + categorical_features
target = 'target_comprou'

X = df[features]
y = df[target]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf_xgb = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
)
model_pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf_xgb)
])
model_pipeline_xgb.fit(X_train, y_train)
y_pred_rf = model_pipeline_xgb.predict(X_test)

preprocessor_xgb = model_pipeline_xgb.named_steps['preprocessor']
xgb_model = model_pipeline_xgb.named_steps['classifier']
cat_feature_names = preprocessor_xgb.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)
importances = xgb_model.feature_importances_
insights_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.set_page_config(
    page_title="Dashboard GoCâmbio - Análise de Campanhas e Vendas",
    layout="wide",
    initial_sidebar_state="collapsed"
)

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
def format_currency(value):
    try:
        return locale.currency(value, grouping=True)
    except:
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

aba1, aba2, aba3 = st.tabs(["📊 Resumo Geral", "📈 Análises Agregadas", "🤖 Insights do Modelo"])

# ==========================================================
# 📊 ABA 1 - RESUMO GERAL
# ==========================================================
with aba1:
    st.title("📊 Resumo Geral das Campanhas e Vendas")

    col1, col2, col3 = st.columns(3)
    with col1:
        total_vendas = df["target_comprou"].sum()
        st.metric("Total de Compras", f"{total_vendas:,}".replace(",", "."))
    with col2:
        total_faturamento = df["totalAmount"].sum()
        st.metric("Faturamento Total", format_currency(total_faturamento))
    with col3:
        total_envios = df["sendAt_week"].count()
        st.metric("Total de Envios", f"{total_envios:,}".replace(",", "."))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        envios_semana = df["sendAt_week"].value_counts().sort_index()
        fig_semana = px.bar(
            x=envios_semana.index,
            y=envios_semana.values,
            labels={"x": "Semana do Ano", "y": "Total de Envios"},
            title="📦 Total de Envios por Semana",
            color_discrete_sequence=["#0083B8"]
        )
        st.plotly_chart(fig_semana, use_container_width=True)

    with col2:
        envios_dia = df["sendAt_weekday_name"].value_counts()
        fig_dia = px.bar(
            x=envios_dia.index,
            y=envios_dia.values,
            labels={"x": "Dia da Semana", "y": "Total de Envios"},
            title="📅 Total de Envios por Dia da Semana",
            color_discrete_sequence=["#00B894"]
        )
        st.plotly_chart(fig_dia, use_container_width=True)

# ==========================================================
# 📈 ABA 2 - ANÁLISES AGREGADAS
# ==========================================================
with aba2:
    st.title("📈 Desempenho por Campanha, Idade e Gênero")

    st.subheader("🎯 Campanhas")
    campanhas_scaled = campanhas.copy()
    campanhas_scaled['taxa_impacto_%'] = campanhas_scaled['taxa_impacto'] * 100

    fig_camp = px.bar(
        campanhas_scaled.melt(
            id_vars="campaignName",
            value_vars=['clientes_impactados', 'gasto_medio', 'taxa_impacto_%'],
            var_name="Métrica", value_name="Valor"
        ),
        x="campaignName", y="Valor", color="Métrica", barmode="group",
        title="Desempenho por Campanha",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={'Valor': ':.2f'}
    )
    st.plotly_chart(fig_camp, use_container_width=True)

    st.subheader("👤 Idade")
    idade_scaled = idade.copy()
    idade_scaled['taxa_impacto_%'] = idade_scaled['taxa_impacto'] * 100

    fig_idade = px.line(
        idade_scaled.melt(
            id_vars="age",
            value_vars=['clientes_impactados', 'gasto_medio', 'taxa_impacto_%'],
            var_name="Métrica", value_name="Valor"
        ),
        x="age", y="Valor", color="Métrica",
        title="Desempenho por Faixa Etária",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={'Valor': ':.2f'}
    )
    st.plotly_chart(fig_idade, use_container_width=True)

    st.subheader("⚧️ Gênero")
    genero_scaled = genero.copy()
    genero_scaled['taxa_impacto_%'] = genero_scaled['taxa_impacto'] * 100

    fig_genero = px.bar(
        genero_scaled.melt(
            id_vars="gender",
            value_vars=['clientes_impactados', 'gasto_medio', 'taxa_impacto_%'],
            var_name="Métrica", value_name="Valor"
        ),
        x="gender", y="Valor", color="Métrica", barmode="group",
        title="Desempenho por Gênero",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={'Valor': ':.2f'}
    )
    st.plotly_chart(fig_genero, use_container_width=True)

# ==========================================================
# 🤖 ABA 3 - INSIGHTS DO MODELO
# ==========================================================
with aba3:
    st.title("🤖 Variáveis que mais impactam na Compra (XGBoost)")

    st.markdown("As variáveis abaixo representam as **features mais relevantes** para o modelo prever se o cliente realizará uma compra:")

    top_vars = insights_df.head(10)
    fig_imp = px.bar(
        top_vars.sort_values(by="Importance", ascending=True),
        x="Importance", y="Feature",
        orientation="h",
        title="Top 10 Variáveis Mais Importantes",
        color="Importance",
        color_continuous_scale=[
            (0.0, "#A7C7E7"),
            (0.5, "#4682B4"),
            (1.0, "#0A3D62")
        ]
    )
    fig_imp.update_layout(template="simple_white", title_x=0.1)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 🧩 Importâncias Completas")
    st.dataframe(
        insights_df.style.format({"Importance": "{:.4f}"}).background_gradient(cmap="Blues"),
        use_container_width=True,
        hide_index=True
    )
