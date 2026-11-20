import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
import hashlib
import json
from pathlib import Path

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
<h2>🎥 Automated Sperm Motility Analysis</h2>
</div>
""", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload microscopic video",
        type=["mp4", "avi"]
    )

    if uploaded_video:

        video_bytes = uploaded_video.getvalue()
        video_id = hashlib.sha256(video_bytes).hexdigest()[:12]
        enhanced_output_dir = os.path.join("outputs", f"enhanced_{video_id}")

        os.makedirs(enhanced_output_dir, exist_ok=True)
        video_extension = os.path.splitext(uploaded_video.name)[1].lower()
        video_path = os.path.join(
            enhanced_output_dir, f"source{video_extension or '.mp4'}"
        )
        with open(video_path, "wb") as video_file:
            video_file.write(video_bytes)

        st.markdown("### Automated AI Video Analysis")
        st.markdown(
            """
            <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
              <span>Upload</span><b>→</b><span>Preprocessing</span><b>→</b>
              <span>Detection</span><b>→</b><span>Tracking</span><b>→</b>
              <span>Movement features</span><b>→</b><span>Motility report</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        analysis_key = f"enhanced_analysis_{video_id}"
        if analysis_key not in st.session_state:
            from tracking.sperm_tracking_enhanced import DEFAULT_MODEL, run_analysis

            st.markdown("**Live AI detections and tracking IDs**")
            live_preview = st.empty()
            progress_bar = st.progress(0.0, text="Preparing video analysis...")

            def update_analysis_progress(processed_frames, total_frames):
                if total_frames > 0:
                    fraction = min(processed_frames / total_frames, 1.0)
                    progress_bar.progress(
                        fraction,
                        text=(
                            f"Analyzing frame {processed_frames:,} "
                            f"of {total_frames:,}"
                        ),
                    )
                else:
                    progress_bar.progress(
                        0.0,
                        text=f"Analyzing frame {processed_frames:,}",
                    )

            def update_live_preview(frame):
                live_preview.image(
                    frame,
                    channels="RGB",
                    caption="Live tracked sperm IDs",
                    use_container_width=True,
                )

            try:
                with st.spinner(
                    "Preprocessing video, detecting sperm, and extracting tracks..."
                ):
                    st.session_state[analysis_key] = run_analysis(
                        video_path=Path(os.path.abspath(video_path)),
                        model_path=DEFAULT_MODEL,
                        output_dir=Path(os.path.abspath(enhanced_output_dir)),
                        progress_callback=update_analysis_progress,
                        preview_callback=update_live_preview,
                    )
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                progress_bar.empty()
                st.error(f"Video analysis could not be completed: {error}")
                st.stop()
            progress_bar.empty()

        summary = st.session_state[analysis_key]
        metadata_frame_count = summary["metadata_frame_count"]
        frame_count_label = (
            f"{metadata_frame_count:,} frames (metadata)"
            if metadata_frame_count > 0
            else f"{summary['frames_processed']:,} frames processed"
        )
        st.caption(
            f"{summary['width']} × {summary['height']} px · "
            f"{summary['fps']:g} FPS · {frame_count_label} · "
            f"{summary['measurement_units']}"
        )

        st.markdown("---")
        st.subheader("Detection & tracking")
        detection_col, tracked_col, consistency_col = st.columns(3)
        detection_col.metric(
            "Total sperm detections", f"{summary['detections']:,}"
        )
        tracked_col.metric("Successfully tracked", f"{summary['unique_tracks']:,}")
        consistency = summary["tracking_consistency_pct"]
        consistency_col.metric(
            "Tracking consistency",
            f"{consistency:.1f}%" if consistency is not None else "Unavailable",
        )
        st.caption(
            "Tracking consistency is mean frame coverage within each track's observed span."
        )

        def display_metric(value, unit, multiplier=1.0, suffix=""):
            if value is None:
                return "Not measurable"
            return f"{value * multiplier:.2f} {unit}{suffix}"

        average_metrics = summary["average_metrics"]
        st.markdown("---")
        st.subheader("Movement analysis")
        movement_columns = st.columns(4)
        movement_columns[0].metric(
            "Curvilinear velocity (VCL)",
            display_metric(average_metrics["VCL_px_s"], "px/s"),
        )
        movement_columns[1].metric(
            "Straight-line velocity (VSL)",
            display_metric(average_metrics["VSL_px_s"], "px/s"),
        )
        movement_columns[2].metric(
            "Average path velocity (VAP)",
            display_metric(average_metrics["VAP_px_s"], "px/s"),
        )
        movement_columns[3].metric(
            "Linearity (LIN)",
            display_metric(average_metrics["LIN"], "%", 100.0),
        )
        movement_columns = st.columns(3)
        movement_columns[0].metric(
            "Straightness (STR)",
            display_metric(average_metrics["STR"], "%", 100.0),
        )
        movement_columns[1].metric(
            "Amplitude of lateral head displacement (ALH)",
            display_metric(average_metrics["ALH_px"], "px"),
        )
        movement_columns[2].metric(
            "Beat-cross frequency (BCF)",
            display_metric(average_metrics["BCF_hz_estimate"], "Hz"),
        )

        st.markdown("---")
        st.subheader("Motility classification")
        percentages = summary["motility_percentages"]
        counts = summary["classification_counts"]
        motility_columns = st.columns(4)
        for column, label, key in zip(
            motility_columns,
            ("Progressive", "Non-progressive", "Immotile", "Total motile"),
            ("Progressive", "Non-Progressive", "Immotile", "Total motile"),
        ):
            percentage = percentages[key]
            count = (
                counts["Progressive"] + counts["Non-Progressive"]
                if key == "Total motile"
                else counts[key]
            )
            column.metric(
                label,
                f"{percentage:.1f}%" if percentage is not None else "Unavailable",
                f"{count:,} tracks",
            )

        st.bar_chart(
            pd.DataFrame(
                {
                    "Sperm tracks": {
                        "Progressive": counts["Progressive"],
                        "Non-progressive": counts["Non-Progressive"],
                        "Immotile": counts["Immotile"],
                    }
                }
            )
        )
        st.caption(
            "Percentages are based on tracks classified for motility."
        )

        st.markdown("---")
        st.subheader("Video review")
        original_column, annotated_column = st.columns(2)
        with original_column:
            st.markdown("**Original microscopic video**")
            st.video(video_bytes)

        enhanced_video_path = summary["outputs"]["video"]
        with annotated_column:
            st.markdown("**AI detections, tracking IDs & trajectories**")
            st.caption(
                "Green: progressive · Amber: non-progressive · "
                "Red: immotile · Gray: insufficient data"
            )
            with open(enhanced_video_path, "rb") as video_file:
                annotated_video = video_file.read()
            st.video(annotated_video)
            st.download_button(
                "Download annotated video",
                data=annotated_video,
                file_name="sperm_tracking_analysis.mp4",
                mime="video/mp4",
            )

        tracks_path = summary["outputs"]["tracks_csv"]
        tracks_df = pd.read_csv(tracks_path)
        if not tracks_df.empty:
            st.markdown("---")
            st.subheader("Movement visualizations")
            track_ids = tracks_df["track_id"].value_counts().head(15).index
            visible_tracks = tracks_df[
                tracks_df["track_id"].isin(track_ids)
            ].copy()
            trajectory_chart = visible_tracks.pivot_table(
                index="time_s",
                columns="track_id",
                values="x_px",
                aggfunc="first",
            )
            trajectory_chart.columns = [
                f"ID {track_id}" for track_id in trajectory_chart.columns
            ]
            st.markdown("**Horizontal position by time (top 15 longest tracks)**")
            st.line_chart(trajectory_chart)

            visible_tracks = visible_tracks.sort_values(["track_id", "frame"])
            grouped_tracks = visible_tracks.groupby("track_id")
            dx = grouped_tracks["x_px"].diff()
            dy = grouped_tracks["y_px"].diff()
            elapsed_s = grouped_tracks["frame"].diff() / summary["fps"]
            visible_tracks["speed_px_s"] = (
                np.sqrt(dx.pow(2) + dy.pow(2)) / elapsed_s.replace(0, np.nan)
            )
            velocity_chart = visible_tracks.pivot_table(
                index="time_s",
                columns="track_id",
                values="speed_px_s",
                aggfunc="first",
            )
            velocity_chart.columns = [
                f"ID {track_id}" for track_id in velocity_chart.columns
            ]
            st.markdown("**Instantaneous tracked velocity (px/s)**")
            st.line_chart(velocity_chart)

        heatmap_path = summary["outputs"]["heatmap"]
        if os.path.exists(heatmap_path):
            st.markdown("**Movement density**")
            st.image(heatmap_path)

        st.markdown("---")
        st.subheader("Final analysis report")
        metrics_path = summary["outputs"]["metrics_csv"]
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            st.info("No classified sperm tracks are available for this report.")
        else:
            report_metrics = metrics_df.loc[
                metrics_df["classification"] != "Insufficient data"
            ].drop(
                columns=[
                    "classification_method",
                    "progressive_threshold_px_s",
                    "motile_threshold_px_s",
                ],
                errors="ignore",
            )
            report_metrics["measurement_units"] = report_metrics[
                "measurement_units"
            ].replace("pixel-based (uncalibrated)", "pixel-based")
            st.dataframe(
                report_metrics,
                use_container_width=True,
            )
        report_summary = summary.copy()
        report_summary.pop("classification_method", None)
        report_summary["classification_counts"] = {
            category: summary["classification_counts"].get(category, 0)
            for category in ("Progressive", "Non-Progressive", "Immotile")
        }
        report_summary["measurement_units"] = summary["measurement_units"].replace(
            "pixel-based (uncalibrated)", "pixel-based"
        )
        st.download_button(
            "Download analysis summary (JSON)",
            data=json.dumps(report_summary, indent=2, ensure_ascii=False),
            file_name="sperm_motility_analysis_summary.json",
            mime="application/json",
        )


