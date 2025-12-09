import soundfile as sf
import io
import numpy as np
import streamlit as st
import joblib
import librosa
from streamlit_mic_recorder import mic_recorder



@st.cache_resource
def load_artifacts():
    model = joblib.load("MLModel.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return model, scaler, label_encoder


model, scaler, label_encoder = load_artifacts()


def extract_features_from_audio_bytes(audio_bytes, target_sr=16000, n_mfcc=40):
    """
    Convert raw audio bytes from the microphone into a 240-dim feature vector.

    Assumes MFCC + delta + delta-delta with mean and std over time:
      40 MFCC
      40 delta
      40 delta-delta
    → 120 coefficients
    → mean + std → 240 features total
    """
    # 1) Try to read the bytes as audio using soundfile
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # If stereo, average to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    y = data

    # 2) If sample rate doesn't match training SR, resample with librosa
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 3) 25 ms window, 10 ms hop (same as RAVDESS pipeline)
    n_fft = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    all_feats = np.concatenate([mfcc, delta, delta2], axis=0)  # (120, T)

    mean_feats = all_feats.mean(axis=1)
    std_feats = all_feats.std(axis=1)
    clip_features = np.concatenate([mean_feats, std_feats], axis=0)  # (240,)

    return clip_features.astype(np.float32)



def predict_emotion_from_bytes(audio_bytes):
    features = extract_features_from_audio_bytes(audio_bytes)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    probs = model.predict_proba(features_scaled)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return pred_label, probs


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="RAVDESS Emotion Detector", page_icon="🎙")
st.title("🎙 RAVDESS Emotion Detector")
st.write(
    "Say **'Kids are talking by the door'** into your microphone with any emotion "
    "(happy, sad, angry, etc.), and the model will predict your emotion."
)

st.markdown(
    """**How to use:**
1. Click **Start recording**
2. Say the sentence with your chosen emotion
3. Click **Stop recording**
4. Click **Analyze Emotion** to see the prediction
"""
)

if mic_recorder is None:
    st.stop()

audio = mic_recorder(
    start_prompt="🎧 Start recording",
    stop_prompt="🛑 Stop recording",
    just_once=False,          # keep this so audio persists on rerun
    use_container_width=True,
    key="recorder",
    format="wav",             #IMPORTANT
)

if audio is not None:
    st.write("Audio keys:", list(audio.keys()))
    st.audio(audio["bytes"], format="audio/wav")

    if st.button("Analyze Emotion"):
        with st.spinner("Analyzing..."):
            pred_label, probs = predict_emotion_from_bytes(audio["bytes"])

        st.subheader(f"Predicted emotion: **{pred_label}**")

        class_names = list(label_encoder.classes_)
        st.write("Class probabilities:")

        st.dataframe(
            {
                "emotion": class_names,
                "probability": [float(round(p, 3)) for p in probs],
            }
        )
