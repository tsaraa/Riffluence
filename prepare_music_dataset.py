import pandas as pd

# Load dataset
df = pd.read_csv("responses.csv")

# Columns to keep
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

cols = ["Age", "Gender"] + music_cols

df = df[cols]

print(df.head())
print(df.shape)

df = df.dropna()

print("After cleaning:", df.shape)

df["Gender"] = df["Gender"].map({
    "male": 0,
    "female": 1
})

print(df["Gender"].value_counts())

df.to_csv(
    "music_preferences_clean.csv",
    index=False
)

print("Dataset saved.")