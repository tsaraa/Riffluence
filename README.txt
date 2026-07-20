=========================================
Riffluence
Facial-Based Music Recommendation System
=========================================

Author: Amira Tsara Bt Mohd Zamri


=========================================
Project Description
=========================================

This project is a facial-based music recommendation system that predicts a user's age and gender from a facial image using an EfficientNetB0 deep learning model. The predicted demographic information is then used by a Random Forest recommendation model to predict music preferences and recommend songs using the Spotify API.


=========================================
Development Environment
=========================================

Operating System: Windows

Programming Language: Python

Framework: Streamlit

IDE: Visual Studio Code


=========================================
Required Python Libraries
=========================================

Open a terminal and install all required packages using: pip install -r requirements.txt

If needed, install the following manually:

pip install streamlit
pip install tensorflow
pip install tf-keras
pip install deepface
pip install retina-face
pip install opencv-python
pip install pillow
pip install numpy
pip install pandas
pip install scikit-learn
pip install spotipy
pip install joblib


=========================================
Datasets
=========================================

UTKFace Dataset: Used for training the age and gender prediction model.

Download: https://www.kaggle.com/datasets/jangedoo/utkface-new?select=UTKFace

Use the utkface_aligned_cropped/UTKFace folder and place it in the project folder.

-----------------------------------------

Young People Survey Dataset: Used for training the music recommendation model.

Download: https://www.kaggle.com/datasets/miroslavsabo/young-people-survey

Extract the file and use: responses.csv (this file is already in the project folder). Put it in the project folder.

Run: prepare_music_dataset.py (Command : python -m streamlit run prepare_music_dataset.py)

This generates: music_preferences_clean.csv (this file is already in the project folder)


=========================================
Training Models
=========================================

Train Age & Gender Model

Command: python train_age_gender_model.py

Output: age_gender_model.keras

This will take some time.

-----------------------------------------

Train Music Recommendation Model

Command: python train_genre_model.py

Output: music_recommender.pkl (this file is already in the project folder)


=========================================
Running the System
=========================================

Run the command: python -m streamlit run riffluence.py

The application will open automatically in your web browser.