import streamlit as st
import cv2
import numpy as np
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from deepface import DeepFace

import joblib

@st.cache_resource
def load_music_model():
    return joblib.load(
        "music_recommender.pk"
    )

music_model = load_music_model()

# ==================================================
# SPOTIFY GENRE MAPPING
# ==================================================

spotify_genre_map = {

    "Dance": "dance",

    "Classical music": "classical",

    "Pop": "pop",

    "Rock": "rock",

    "Metal or Hardrock": "metal",

    "Hiphop, Rap": "hip-hop",

    "Swing, Jazz": "jazz",

    "Techno, Trance": "electronic",

    "Alternative": "alternative",

    "Country": "country"
}

# ==================================================
# PAGE SETTINGS
# ==================================================
st.set_page_config(
    page_title="Facial Music Recommender",
    layout="centered"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #fff0f6;
}

/* Headers */
h1, h2, h3 {
    color: #d63384 !important;
}

/* Normal text */
p, label, div {
    color: #4a4a4a;
}

/* Success boxes */
div[data-testid="stAlert"] {
    background-color: #ffd6e8;
    color: #d63384;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background-color: #ff69b4;
    color: white;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #ff1493;
}

/* Radio buttons */
div[role="radiogroup"] label {
    color: #d63384 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffe4f2;
    border-radius: 10px;
    padding: 10px;
}

/* Metric / prediction cards */
.pred-card {
    background: linear-gradient(
        135deg,
        #ff69b4,
        #ffb6c1
    );
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 20px;
    font-weight: bold;
    min-height: 70px;
    display: flex;
    align-items: center;
}
            
div[data-testid="stLinkButton"] a {
    background-color: #ff1493 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: bold !important;
}

div[data-testid="stLinkButton"] a:hover {
    background-color: #ff69b4 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🎵 Riffluence")
st.write("Upload a facial image to get music recommendations")

# ==================================================
# SPOTIFY CONFIG
# ==================================================
import os

CLIENT_ID = "06c80b6a31744b7b8850be56a04c0c0c"
CLIENT_SECRET = "4b641a612f1a4929a637af761e2e28b2"

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

# ==================================================
# LOAD MODEL
# ==================================================
from keras.models import load_model

@st.cache_resource
def load_age_gender_model():
    return load_model(
        "age_gender_model.keras",
        compile=False,
        safe_mode=False
    )

model = load_age_gender_model()

# ==================================================
# PREDICT AGE + GENDER
# ==================================================
def predict_age_gender(face):

    from tensorflow.keras.applications.efficientnet import preprocess_input

    face = cv2.resize(face, (224,224))

    face = face.astype(np.float32)

    face = preprocess_input(face)

    face = np.expand_dims(face, axis=0)

    preds = model.predict(
        face,
        verbose=0
    )

    gender_pred = preds[0][0]
    age_pred = preds[1][0]

    # Highest probability
    gender_conf = float(np.max(gender_pred))

    gender_index = np.argmax(gender_pred)

    if gender_conf < 0.60:

        gender = "Uncertain"

    else:

        gender = (
            "Male"
            if gender_index == 0
            else "Female"
        )

    # -----------------------
    # Exact Age
    # -----------------------

    predicted_age = int(round(age_pred[0]))

    predicted_age = max(
        1,
        min(predicted_age, 100)
    )

    if predicted_age <= 12:
        age_group = "Child"

    elif predicted_age <= 24:
        age_group = "Young Adult"

    elif predicted_age <= 59:
        age_group = "Adult"

    else:
        age_group = "Older Adult"

    return (
        predicted_age,
        age_group,
        gender,
        gender_conf
    )

def predict_music_preferences(age, gender):

    if gender == "Male":
        gender_numeric = 0
    elif gender == "Female":
        gender_numeric = 1
    else:
        gender_numeric = 0.5

    features = [[
        age,
        gender_numeric
    ]]

    scores = music_model.predict(
        features
    )[0]

    genres = [

        "Dance",
        "Classical music",
        "Pop",
        "Rock",
        "Metal or Hardrock",
        "Hiphop, Rap",
        "Swing, Jazz",
        "Techno, Trance",
        "Alternative",
        "Country"

    ]

    genre_scores = dict(
        zip(genres, scores)
    )

    sorted_genres = sorted(
        genre_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )  

    return sorted_genres[:3]

# ==================================================
# SPOTIFY SONG RECOMMENDATION
# ==================================================

@st.cache_data(ttl=3600)
def recommend_music_from_genres(top_genres):

    songs = []

    for genre, score in top_genres:

        spotify_genre = spotify_genre_map.get(
            genre,
            genre.lower()
        )

        try:
            results = sp.search(
                q=f'genre:{spotify_genre}',
                type="track",
                limit=10
            )
        except Exception:
            continue

        for track in results["tracks"]["items"]:

            songs.append({

                "song": track["name"],

                "artist": track["artists"][0]["name"],

                "url": track["external_urls"]["spotify"],

                "image": track["album"]["images"][0]["url"],

                "genre": genre

            })

    return songs

# ==================================================
# IMAGE INPUT OPTIONS
# ==================================================

st.subheader("📸 Choose Image Input")

input_option = st.radio(
    "",
    ["Upload Image", "Use Camera"]
)

uploaded_file = None
camera_image = None

# --------------------------------
# Upload Image
# --------------------------------
if input_option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

# --------------------------------
# Camera Capture
# --------------------------------
else:

    camera_image = st.camera_input(
        "Take a Picture"
    )

# ==================================================
# MAIN SYSTEM
# ==================================================
if uploaded_file is not None or camera_image is not None:

    with st.spinner("Analyzing face and generating recommendations..."):

        # =================================
        # READ IMAGE
        # =================================
        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")

        else:

            image = Image.open(camera_image).convert("RGB")

        # Convert PIL → NumPy
        img = np.array(image)

        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=False
        )

        st.write(f"Faces detected: {len(faces)}")

        if len(faces) == 0:

            st.warning("⚠ No face detected.")

        else:

            face_conf = faces[0]["confidence"] * 100

            if face_conf >= 50:
                st.write(
                    f"Face Detection Confidence: {face_conf:.2f}%"
                )

            face = faces[0]["face"]
            face = (face * 255).astype(np.uint8)

            facial_area = faces[0]["facial_area"]

            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]

            # =================================
            # DISPLAY COLUMNS
            # =================================
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                        face,
                        caption="Detected Face",
                        width=250
                    )

                # =================================
                # PREDICTION
                # =================================
                try:

                    age, age_group, gender, gender_conf = predict_age_gender(face)

                    # Draw box
                    cv2.rectangle(
                        img,
                        (x, y),
                        (x+w, y+h),
                        (0,255,0),
                        2
                    )

                    # Draw label
                    label = f"{gender}, {age} yrs"

                    cv2.putText(
                        img,
                        label,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )

                    # =================================
                    # RIGHT COLUMN
                    # =================================
                    with col2:

                        st.markdown(
                            f'<div class="pred-card">Age: {age}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="pred-card">Age Group: {age_group}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="pred-card">Gender: {gender}</div>',
                            unsafe_allow_html=True
                        )

                        if gender != "Uncertain":

                            st.metric(
                                "Gender Confidence",
                                f"{gender_conf:.2%}"
                            )

                        else:

                            st.warning(
                                "Gender could not be confidently determined."
                            )

                    # =================================
                    # MUSIC RECOMMENDATION
                    # =================================
                    top_genres = predict_music_preferences(
                        age,
                        gender
                    )

                    top_genres = [
                        (str(g), float(s))
                        for g, s in top_genres
                    ]

                    songs = recommend_music_from_genres(
                        top_genres
                    )

                    genres = [
                        g[0]
                        for g in top_genres
                    ]

                    # =================================
                    # SONG DISPLAY
                    # =================================
                    
                    st.subheader(
                        "🎵 Predicted Music Preferences"
                    )

                    for genre, score in top_genres:

                        normalized = min(
                            max(score / 5.0, 0),
                            1
                        )

                        st.progress(normalized)

                        st.write(
                            f"{genre}: {score:.2f}/5"
                        )

                    for genre in genres:

                        st.subheader(f"🎶 {genre}")

                        genre_songs = [
                            s for s in songs
                            if s.get("genre") == genre
                        ]

                        cols = st.columns(3)

                        for i, song in enumerate(genre_songs):

                            with cols[i % 3]:

                                st.image(
                                    song["image"],
                                    use_container_width=True
                                )

                                st.caption(song["song"])

                                st.write(song["artist"])

                                st.link_button(
                                    "Spotify",
                                    song["url"]
                                )

                    # =================================
                    # SUCCESS EFFECT
                    # =================================
                    

                except Exception as e:

                    st.error("Prediction failed.")
                    st.text(str(e))

            # =================================
            # SHOW FINAL IMAGE
            # =================================
            with st.expander("View Detection Result"):
                st.image(
                    img,
                    use_container_width=True
                )

            st.markdown("---")
            st.caption(
                "Riffluence • Facial-Based Music Recommendation System • FYP 2026"
                )