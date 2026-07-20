import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("music_preferences_clean.csv")

# =====================================
# INPUT FEATURES
# =====================================

X = df[["Age", "Gender"]]

# =====================================
# TARGETS
# =====================================

music_cols = [
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

y = df[music_cols]

# =====================================
# SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# MODEL
# =====================================

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
)

# =====================================
# TRAIN
# =====================================

model.fit(
    X_train,
    y_train
)

# =====================================
# EVALUATE
# =====================================

preds = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    preds
)

print("MAE:", mae)

# =====================================
# SAVE
# =====================================

joblib.dump(
    model,
    "music_recommender.pk"
)

print("Model Saved.")