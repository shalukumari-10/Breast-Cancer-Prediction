# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Try to import fancy nav; if missing, we'll use a simple sidebar radio
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# CONFIG -
st.set_page_config(
    page_title="OncoPlus – AI Powered Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
)

# THEME / CSS -
st.markdown(
    """
    <style>

    .stApp { background-color: #e8f1fc; }

    /*  STICKY TITLE  */
    .title-header {
        position: sticky;
        top: 0;
        z-index: 9999;
        background: linear-gradient(90deg, #a8d0e6, #234a7b);
        padding: 15px 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        color: white;
    }

    .subtitle {
        position: sticky;
        top: 70px; 
        z-index: 9998;
        background: #e8f1fc;
        padding: 8px 0;
        text-align:center;
        font-size:18px;
        color: #234a7b;
    }

    .sidebar .sidebar-content {
        background-color: #e9f6ff;
        padding-top: 20px;
    }

    .risk-circle {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:28px;
        font-weight:700;
        margin: 20px auto;
        color: #0b2447;
    }

    .card {
        background: white;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }

    .nav-label { font-weight:600; color:#08306b; margin-bottom:10px;}

    /* OPTION MENU SPACING */
    div[data-testid="stSidebar"] div[class*="option_menu"] a {
        padding-top: 22px !important;
        padding-bottom: 22px !important;
        margin: 16px 0 !important;
        display: block !important;
        font-size: 18px !important;
    }

    /* Make sidebar taller */
    div[data-testid="stSidebar"] {
        padding-top: 40px !important;
        padding-bottom: 60px !important;
        min-height: 100vh;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


#  MODEL FILE PATHS -
MODEL_PATH = "logistic_model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None
model_load_error = None
scaler_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model_load_error = e

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    scaler_load_error = e

#  NAVIGATION -
menu_options = ["Home", "Predictor", "About Model", "Graphs & Insights", "Contact", "Disclaimer"]

if HAS_OPTION_MENU:
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=["house", "activity", "info-circle", "bar-chart", "envelope", "exclamation-triangle"],
            menu_icon="app",
            default_index=1,
            orientation="vertical",
            styles={
                "container": {"padding": "5px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "padding": "8px 12px"},
                "nav-link-selected": {"background-color": "#4a90e2", "color": "white"},
            }
        )
else:
    with st.sidebar:
        st.markdown("<div class='nav-label'>Menu</div>", unsafe_allow_html=True)
        selected = st.radio("", menu_options)

#  HEADER -
st.markdown("<div class='title-header'>OncoPlus – AI Powered Breast Cancer Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Quick screening tool • Not a medical diagnosis</div>", unsafe_allow_html=True)
st.write("---")

#  FEATURES -
FEATURE_NAMES = [
    "radius_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "texture_se",
    "perimeter_se",
    "radius_worst",
    "texture_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst"
]

FEATURE_RANGES = {
    "radius_mean": (6.981, 28.110),
    "compactness_mean": (0.01938, 0.34540),
    "concavity_mean": (0.00000, 0.42680),
    "concave_points_mean": (0.00000, 0.20120),
    "symmetry_mean": (0.10600, 0.30400),
    "texture_se": (0.36020, 4.88500),
    "perimeter_se": (0.75700, 21.98000),
    "radius_worst": (7.93000, 33.13000),
    "texture_worst": (12.02000, 49.54000),
    "smoothness_worst": (0.07117, 0.22260),
    "compactness_worst": (0.02729, 1.05800),
    "concavity_worst": (0.00000, 1.25200),
    "concave_points_worst": (0.00000, 0.29100),
    "symmetry_worst": (0.15650, 0.66380),
}

# Helper to format feature names nicely
def format_feature_name(name):
    return name.replace("_", " ").title()

#  PAGES -
if selected == "Home":
    st.header("🏠 Welcome to OncoPlus")
    st.markdown(
        """
        **OncoPlus** is an AI-powered early screening assistant for breast cancer risk based on 14 diagnostic features.
        Use the **Predictor** page to input features and get a risk percentage.
        """
    )
    st.info("This app is for educational and screening purposes only. Always consult a healthcare professional for a clinical diagnosis.")

elif selected == "Predictor":
    st.header("🔬 Predictor")

    if model_load_error:
        st.error(f"Failed to load model: {model_load_error}")
        st.stop()
    if scaler_load_error:
        st.error(f"Failed to load scaler: {scaler_load_error}")
        st.stop()

    st.markdown("Enter the 14 diagnostic features below using sliders:")

    n_cols = 2
    col_list = st.columns(n_cols)
    values = []  # IMPORTANT

    for i, fname in enumerate(FEATURE_NAMES):
        col_idx = i % n_cols
        with col_list[col_idx]:

            _, max_val = FEATURE_RANGES[fname]

            slider_val = st.slider(
                format_feature_name(fname),
                min_value=0.0,
                max_value=float(max_val),
                value=0.0,
                step=0.0001,
                key=f"slider_{fname}"
            )

            values.append(slider_val)   # <-- Correct indentation

    inputs = np.array(values).reshape(1, -1)

    if st.checkbox("Show input as DataFrame"):
        st.dataframe(
            pd.DataFrame(
                inputs,
                columns=[format_feature_name(f) for f in FEATURE_NAMES]
            )
        )

    if st.button("🔎 Predict Risk"):
        try:
            scaled = scaler.transform(inputs)
            prob = model.predict_proba(scaled)[0][1]

            # Status logic
            if prob >= 0.65:
                color = "#d9534f"
                status = "Malignant"
            elif prob >= 0.30:
                color = "#f0ad4e"
                status = "Intermediate / Watch"
            else:
                color = "#5cb85c"
                status = "Benign"

            circle = f"""
            <div style="display:flex;align-items:center;justify-content:center;">
                <div class="risk-circle" style="border:12px solid {color}; color:{color};">
                    {prob*100:.1f}%
                </div>
            </div>
            <div style="text-align:center;font-weight:700;color:{color};font-size:20px">{status}</div>
            """
            st.markdown(circle, unsafe_allow_html=True)
            st.write("---")
            st.write(f"**Probability of malignancy:** {prob:.3f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


elif selected == "About Model":
    st.header("ℹ️ About the Model")
    st.markdown(
        """
        **This model uses **Logistic Regression** trained on 14 selected diagnostic features.**
        """
    )
    
    st.markdown(
        """
        **✅Key Features Used in the Prediction Model**

    The model uses a focused set of features that capture the shape, structure, and texture of breast cell nuclei. 
    These features are highly predictive in distinguishing between benign and malignant tumors.

    ⭐**Mean Features**

    • Radius (mean) – Average distance from the center to the cell perimeter

    • Compactness (mean) – Describes how tightly packed the cell structure is

    • Concavity (mean) – Indicates the severity of concave regions in the cell contour

    • Concave Points (mean) – Number of concave sections along the contour

    • Symmetry (mean) – Measures the symmetry of the cell nucleus

    ⭐**Standard Error Features**

    • Texture (SE) – Variation in pixel intensity values

    • Perimeter (SE) – Variation in perimeter measurements

    ⭐**Worst (Largest) Features**

    • Radius (worst) – Largest recorded radius measurement

    • Texture (worst) – Maximum variation in texture

    • Smoothness (worst) – Highest variation in radius lengths

    • Compactness (worst) – Maximum compactness level

    • Concavity (worst) – Maximum severity of concave regions

    • Concave Points (worst) – Highest count of concave contour points

    • Symmetry (worst) – Maximum symmetry measurement
        """
    )
    st.image("MODEL_ACC.png", caption="ACCURACY OF DIFFERENT MODELS", use_container_width=False)
    col1, col2 = st.columns(2)

    with col1:
        st.image("confusion_matrix.png", caption="Confusion Matrix of Logistic Regression Model", use_container_width=True)

    with col2:
        st.image("heatmap.png",caption="Correlation Heatmap", use_container_width=True)

elif selected == "Graphs & Insights":
    st.header("📊 Graphs & Insights")
    uploaded = st.file_uploader("Upload CSV with the 14 features", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        st.write("### Feature Distributions")
        for col in FEATURE_NAMES:
            if col in df.columns:
                st.write(f"**{format_feature_name(col)}**")
                st.bar_chart(df[col].fillna(0))

elif selected == "Contact":
    st.header("📩 Contact")
    st.markdown(
        """
        Email: support@oncoplus.ai  
        Phone: +91-XXXXXXXXXX
        """
    )

elif selected == "Disclaimer":
    st.header("⚠️ Disclaimer")
    st.markdown(
        """
        This tool is NOT a clinical diagnostic system.  
        Please consult a licensed medical professional.
        """
    )

st.write("---")
st.markdown("Built with ❤️ • OncoPlus")


