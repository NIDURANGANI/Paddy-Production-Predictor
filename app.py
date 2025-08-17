import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error # type: ignore
import plotly.express as px # pyright: ignore[reportMissingImports]

# ========================
# Page Config
# ========================
st.set_page_config(
    page_title="🌾 Agricultural Production Predictor",
    page_icon="🌱",
    layout="wide"
)

# ========================
# Dark Mode Toggle
# ========================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

dark_css = """
    <style>
    body {background-color: #121212; color: #ffffff;}
    .stApp {background: #121212; color: #ffffff;}
    .stButton>button {background-color: #1f7a1f; color: white; border-radius: 10px;}
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #333333; color: #ffffff;
    }
    .stDataFrame, .stMarkdown, .stPlotlyChart {
        background: #1e1e1e; border-radius: 10px; padding: 10px;
    }
    </style>
"""

light_css = """
    <style>
    body {background-color: #f5fff5;}
    .stButton>button {background-color: #2e8b57; color: white; border-radius: 10px;}
    </style>
"""

st.sidebar.title("⚙️ Settings")
st.sidebar.button(
    "🌙 Toggle Dark Mode" if not st.session_state.dark_mode else "☀️ Toggle Light Mode",
    on_click=toggle_dark_mode
)
st.markdown(dark_css if st.session_state.dark_mode else light_css, unsafe_allow_html=True)

# ========================
# Load Dataset
# ========================
df = pd.read_csv("2023 Yala.csv")
df.replace(["-", "NA", "NaN", ""], np.nan, inplace=True)

for col in df.columns:
    if col != "District":
        df[col] = pd.to_numeric(df[col], errors='coerce')

target_columns = [
    "Total_Production",
    "Average_Yield",
    "Nett_Extent_Harvested",
    "Rainfed_Yield",
    "All_Schemes_Harvested"
]

df.dropna(subset=target_columns, inplace=True)

categorical_features = ["District"]
numeric_features = [c for c in df.columns if c not in categorical_features + target_columns]

X = df.drop(columns=target_columns)
y = df[target_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# ========================
# UI
# ========================
st.title("🌾 Agricultural Production Predictor")
st.sidebar.subheader("📌 Select Mode")
mode = st.sidebar.radio("Choose:", ["Manual Input", "CSV Upload", "District Comparison"])

# ========================
# Manual Input Mode
# ========================
if mode == "Manual Input":
    st.subheader("📥 Enter Field Data")
    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox("District", df["District"].unique())
        major_sown = st.number_input("Major Schemes Sown", min_value=0.0, step=1.0)
        minor_sown = st.number_input("Minor Schemes Sown", min_value=0.0, step=1.0)
        rainfed_sown = st.number_input("Rainfed Sown", min_value=0.0, step=1.0)
        all_sown = st.number_input("All Schemes Sown", min_value=0.0, step=1.0)

    with col2:
        major_harvested = st.number_input("Major Schemes Harvested", min_value=0.0, step=1.0)
        minor_harvested = st.number_input("Minor Schemes Harvested", min_value=0.0, step=1.0)
        rainfed_harvested = st.number_input("Rainfed Harvested", min_value=0.0, step=1.0)
        all_harvested = st.number_input("All Schemes Harvested", min_value=0.0, step=1.0)
        major_yield = st.number_input("Major Schemes Yield", min_value=0.0, step=0.1)
        minor_yield = st.number_input("Minor Schemes Yield", min_value=0.0, step=0.1)
        rainfed_yield = st.number_input("Rainfed Yield", min_value=0.0, step=0.1)

    if st.button("🌱 Predict"):
        input_data = pd.DataFrame([{
            "District": district,
            "Major_Schemes_Sown": major_sown,
            "Minor_Schemes_Sown": minor_sown,
            "Rainfed_Sown": rainfed_sown,
            "All_Schemes_Sown": all_sown,
            "Major_Schemes_Harvested": major_harvested,
            "Minor_Schemes_Harvested": minor_harvested,
            "Rainfed_Harvested": rainfed_harvested,
            "All_Schemes_Harvested": all_harvested,
            "Major_Schemes_Yield": major_yield,
            "Minor_Schemes_Yield": minor_yield,
            "Rainfed_Yield": rainfed_yield
        }])

        predictions = model.predict(input_data)[0]
        result_df = pd.DataFrame({"Metric": target_columns, "Prediction": predictions})

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df)

        fig = px.bar(result_df, x="Metric", y="Prediction",
                     title="🌱 Predicted Agricultural Outputs",
                     text_auto=".2f", color="Metric")
        st.plotly_chart(fig, use_container_width=True)

# ========================
# CSV Upload Mode
# ========================
elif mode == "CSV Upload":
    st.subheader("📤 Upload CSV for Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        # Clean placeholders → NaN, then coerce numerics
        user_df.replace(["-", "NA", "NaN", ""], np.nan, inplace=True)
        for c in user_df.columns:
            if c != "District":
                user_df[c] = pd.to_numeric(user_df[c], errors="coerce")

        st.write("📄 Cleaned Preview")
        st.dataframe(user_df.head())

        try:
            preds = model.predict(user_df)
            pred_df = pd.DataFrame(preds, columns=target_columns)
            st.subheader("📊 Prediction Results")
            st.dataframe(pred_df)

            avg = pred_df.mean().reset_index()
            avg.columns = ["Metric", "Average Prediction"]
            fig = px.bar(avg, x="Metric", y="Average Prediction",
                         title="🌱 Average Predictions (from CSV)",
                         text_auto=".2f", color="Metric")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇️ Download Predictions as CSV",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Error processing CSV: {e}")

# ========================
# District Comparison Mode (with Tabs + Pie metric selector)
# ========================
elif mode == "District Comparison":
    st.subheader("🏞 Compare District Predictions")

    # Sidebar metric selector for pie
    pie_metric = st.sidebar.selectbox(
        "📊 Pie Chart Metric",
        target_columns,
        index=0
    )

    selected_districts = st.multiselect(
        "Select Districts",
        df["District"].unique()
    )

    if selected_districts:
        subset = df[df["District"].isin(selected_districts)]
        preds = model.predict(subset.drop(columns=target_columns))
        pred_df = pd.DataFrame(preds, columns=target_columns)
        pred_df["District"] = subset["District"].values

        st.dataframe(pred_df)

        tab_pie, tab_breakdown = st.tabs(["🥧 Pie (% Share)", "📈 Breakdown"])

        with tab_pie:
            pie_df = pred_df.groupby("District")[pie_metric].sum().reset_index()
            fig_pie = px.pie(
                pie_df,
                names="District",
                values=pie_metric,
                title=f"🍚 District Contribution to {pie_metric}",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab_breakdown:
            melted = pred_df.melt(id_vars="District", var_name="Metric", value_name="Prediction")
            fig_line = px.line(
                melted, x="Metric", y="Prediction", color="District",
                markers=True, title="📈 District Comparison of Predictions"
            )
            st.plotly_chart(fig_line, use_container_width=True)

# ========================
# Model Performance
# ========================
st.subheader("📈 Model Performance on Test Data")
y_pred = model.predict(X_test)
perf = []
for i, col in enumerate(target_columns):
    perf.append({
        "Metric": col,
        "R²": round(r2_score(y_test[col], y_pred[:, i]), 3),
        "MAE": round(mean_absolute_error(y_test[col], y_pred[:, i]), 2)
    })
st.dataframe(pd.DataFrame(perf))
