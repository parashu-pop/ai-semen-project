import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import subprocess
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="MedAI Fertility Intelligence",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<div style='text-align:center;padding:20px'>
<h1>🧬 MedAI Fertility Intelligence</h1>
<h4>AI-Powered Semen Quality Assessment System</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():

    image_model = models.resnet50(weights=None)
    image_model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(image_model.fc.in_features, 3)
    )
    image_model.load_state_dict(torch.load("backend/models/best_model2.pth", map_location="cpu"))
    image_model.eval()

    lifestyle_model = joblib.load("backend/models/lifestyle_model.pkl")


    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    return (
    image_model,
    lifestyle_model,
    tokenizer,
    llm_model
)


(
    image_model,
    lifestyle_model,
    tokenizer,
    llm_model
) = load_models()

st.markdown("""
<style>

/* Main page */
.stApp {
    background: linear-gradient(
        135deg,
        #f4f8fb 0%,
        #eaf3ff 100%
    );
}

/* Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
    border: none;
    background: linear-gradient(
        90deg,
        #1f77b4,
        #4ea5ff
    );
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 15px;
    padding: 20px;
    border: 1px solid #dce7f3;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Headers */
h1 {
    color: #0f4c81;
    font-weight: 700;
}

h2 {
    color: #0f4c81;
}

h3 {
    color: #0f4c81;
}

/* Cards spacing */
div[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 15px;
    padding: 10px;
    border: 1px solid #dce7f3;
}

/* Success boxes */
.stSuccess {
    border-radius: 12px;
}

/* Warning boxes */
.stWarning {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)
# ================= IMAGE =================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(img):
    img = transform(img).unsqueeze(0)
    outputs = image_model(img)
    _, pred = torch.max(outputs, 1)
    return ["abnormal", "non_sperm", "normal"][pred.item()]

# ================= LIFESTYLE =================
def predict_lifestyle(data):
    x = np.array([[
        data["age"], 0,
        1 if data["smoking"]=="Yes" else 0,
        1 if data["alcohol"]=="Yes" else 0,
        0 if data["exercise"]=="Low" else 1 if data["exercise"]=="Moderate" else 2,
        data["sleep"], data["bmi"]
    ]])
    pred = lifestyle_model.predict(x)[0]
    return ["Low","Medium","High"][pred]

# ================= RULE BASED ADVICE =================
def lifestyle_advice_rules(data):
    advice = []

    if data["sleep"] < 7:
        advice.append("- Improve sleep to 7–8 hours")

    if data["bmi"] >= 25:
        advice.append("- Reduce weight with proper diet")

    if data["smoking"] == "Yes":
        advice.append("- Stop smoking")

    if data["alcohol"] == "Yes":
        advice.append("- Reduce alcohol intake")

    if data["exercise"] == "Low":
        advice.append("- Increase daily physical activity")

    return advice

# ================= IMAGE ADVICE =================
def image_advice(result):
    if result == "normal":
        return [
            "- Maintain current healthy lifestyle",
            "- Continue balanced diet",
            "- Avoid excessive heat exposure"
        ]
    elif result == "abnormal":
        return [
            "- Improve diet with antioxidants",
            "- Reduce stress",
            "- Avoid smoking and alcohol",
            "- Consult doctor if needed"
        ]
    else:
        return [
            "- Seek medical evaluation",
            "- Follow specialist guidance",
            "- Maintain healthy lifestyle"
        ]

# ================= LLM =================
def generate_advice(result, details):

    prompt = f"""
Male fertility expert.

Result: {result}

Details:
{details}

Give 3 short bullet points.
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_length=100, temperature=0.3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================= FINAL SCORE =================
def calculate_final_score(image_result, lifestyle):

    score = 100

    if image_result == "normal":
        score -= 10
    elif image_result == "abnormal":
        score -= 40
    else:
        score -= 70

    if lifestyle == "Low":
        score -= 10
    elif lifestyle == "Medium":
        score -= 25
    else:
        score -= 40

    return max(score, 0)

def interpret(score):
    if score >= 75:
        return "Good Fertility"
    elif score >= 50:
        return "Moderate Fertility"
    else:
        return "Low Fertility"

# ================= MODE =================
with st.sidebar:

    st.title("Navigation")

    mode = st.radio(
        "Choose Analysis",
        [
            "🖼 Image",
            "📋 Lifestyle",
            "🎥 Video"
        ]
    )

    st.markdown("---")

    st.success("🧬 MedAI Dashboard")

st.markdown("""


✅ Image Classification

✅ Lifestyle Risk Analysis

✅ Video Motility Assessment

✅ Fertility Prediction
""")
# ================= IMAGE =================
if mode == "🖼 Image":

    file = st.file_uploader("Upload Image")

    if file:
        img = Image.open(file)
        st.image(img, width=300)

    if st.button("Analyze Image") and file:

        result = predict_image(img)
        st.metric("Sperm Quality", result)

        st.session_state["image"] = result

        st.subheader("Advice")
        for a in image_advice(result):
            st.write(a)

# ================= LIFESTYLE =================
elif mode == "📋 Lifestyle":

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 80)
        height = st.number_input("Height (cm)", 140, 210)
        weight = st.number_input("Weight (kg)", 40, 150)
        sleep = st.slider("Sleep", 1, 12)

    with col2:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        exercise = st.selectbox("Exercise", ["Low", "Moderate", "High"])

    bmi = weight / ((height / 100) ** 2)
    st.write(f"BMI: {bmi:.2f}")

    if st.button("Analyze Lifestyle"):

        level = predict_lifestyle({
            "age": age,
            "smoking": smoking,
            "alcohol": alcohol,
            "bmi": bmi,
            "sleep": sleep,
            "exercise": exercise
        })

        st.metric("Risk", level)
        st.session_state["life"] = level

        # Positive & Negative
        colp, coln = st.columns(2)

        with colp:
            st.subheader("Positive")

            if smoking == "No":
                st.write("✔ Non-smoker")

            if alcohol == "No":
                st.write("✔ No alcohol")

            if sleep >= 7:
                st.write("✔ Good sleep")

            if exercise != "Low":
                st.write("✔ Active")

            if bmi < 25:
                st.write("✔ Healthy BMI")

        with coln:
            st.subheader("Risk")

            if smoking == "Yes":
                st.write("❌ Smoking")

            if alcohol == "Yes":
                st.write("❌ Alcohol")

            if sleep < 7:
                st.write("❌ Poor sleep")

            if exercise == "Low":
                st.write("❌ Low activity")

            if bmi >= 25:
                st.write("❌ High BMI")

        st.subheader("Recommendations")

        rules = lifestyle_advice_rules({
            "sleep": sleep,
            "bmi": bmi,
            "smoking": smoking,
            "alcohol": alcohol,
            "exercise": exercise
        })

        for r in rules:
            st.write(r)

    # ================= FINAL FERTILITY SCORE =================

    st.markdown("---")
    st.header("Final Fertility Score")

    if "image" in st.session_state and "life" in st.session_state:

        score = calculate_final_score(
            st.session_state["image"],
            st.session_state["life"]
        )

        st.metric("Score", f"{score}%")
        st.success(interpret(score))

        st.write(
            f"Image: {st.session_state['image']}"
        )

        st.write(
            f"Lifestyle: {st.session_state['life']}"
        )

    else:

        st.info(
            "Run both Image and Lifestyle analyses to get final result"
        )
   # =========================================================
# VIDEO ANALYSIS
# =========================================================


elif mode == "🎥 Video":

    st.markdown("---")

    st.markdown("""
<div style="
background:white;
padding:15px;
padding:15px;
border-radius:15px;
box-shadow:0px 3px 10px rgba(0,0,0,0.08);
">
<h2>🎥 AI Video Fertility Analysis</h2>
</div>
""", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload Semen Video",
        type=["mp4", "avi"]
    )

    if uploaded_video:

        # SAVE VIDEO
        video_path = "temp_video.mp4"

        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video Uploaded Successfully")

        if st.button("Analyze Video"):

            with st.spinner(
                "Running AI sperm tracking..."
            ):

                subprocess.run([
                    "python",
                    "tracking/sperm_tracking.py"
                ])

                subprocess.run([
                    "python",
                    "tracking/generate_metrics.py"
                ])

            st.success("Tracking Completed")

            # ============================================
            # CHECK CSV
            # ============================================
            if not os.path.exists(
                "outputs/sperm_metrics.csv"
            ):
                st.error("CSV NOT created")
                st.stop()

            st.success("CSV created")

           

            # ============================================
            # LOAD METRICS
            # ============================================
            st.markdown("---")

            st.subheader("📄 Metrics Data")

            metrics_df = pd.read_csv(
                "outputs/sperm_metrics.csv"
            )
            
           

            st.dataframe(metrics_df)

            latest = metrics_df.iloc[-1]

            total_sperm = int(
                latest["total_sperm"]
            )

            progressive_pct = float(
                latest["progressive_motility_pct"]
            )

            non_progressive_pct = float(
                latest["non_progressive_motility_pct"]
            )

            immotile_pct = float(
                latest["immotile_pct"]
            )

            avg_speed = float(
                latest["average_speed"]
            )
            # ============================================
# RULE BASED FERTILITY PREDICTION
# ============================================

            if (
                 progressive_pct >= 40
                and immotile_pct <= 60
                and avg_speed >= 5
                ):

                prediction = "Normal Fertility"

            else:

                prediction = "Abnormal Fertility"
            # ============================================
            # MOTILITY STATISTICS
            # ============================================
            st.markdown("---")

            st.markdown("""
                    <h2 style='text-align:center'>
                    📊 Motility Analysis Dashboard
                    </h2>
                    """, unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Total Sperm",
                total_sperm
            )

            c2.metric(
                "Progressive %",
                f"{progressive_pct:.2f}%"
            )

            c3.metric(
                "Non-Progressive %",
                f"{non_progressive_pct:.2f}%"
            )

            c4.metric(
                "Immotile %",
                f"{immotile_pct:.2f}%"
            )

            c5.metric(
                "Average Speed",
                f"{avg_speed:.2f}"
            )

            # ============================================
            # FINAL PREDICTION
            # ============================================
            st.markdown("---")

            st.subheader(
                "🧠 Final Fertility Prediction"
            )

            if prediction == "Normal Fertility":

                 st.markdown("""
                    <div style="
                     background:#d4edda;
                     padding:25px;
                     border-radius:15px;
                     text-align:center;
                    ">
                    <h2>✅ Normal Fertility</h2>
                    <p>Motility parameters appear healthy.</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:

                 st.markdown("""
                     <div style="
                     background:#f8d7da;
                     padding:25px;
                     border-radius:15px;
                     text-align:center;
                     ">
                    <h2>❌ Abnormal Fertility</h2>
                     <p>Reduced sperm motility detected.</p>
                     </div>
                     """, unsafe_allow_html=True)
            # ============================================
            # SUMMARY
            # ============================================
            st.markdown("---")

            st.subheader(
                "📋 AI Analysis Summary"
            )

            if prediction == "Normal Fertility":

                st.success(
                    """
                    Motility parameters appear normal.
                    Progressive sperm movement is good and
                    fertility potential is likely healthy.
                    """
                )

            else:

                st.warning(
                    """
                    Reduced sperm motility detected.
                    High immotile sperm percentage or low
                    progressive movement may indicate
                    fertility issues.
                    """
                )
                

