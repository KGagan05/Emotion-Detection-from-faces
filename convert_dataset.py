import pandas as pd
import numpy as np
import cv2
import os

data = pd.read_csv("fer2013.csv")

emotions = {
0:"angry",
1:"disgust",
2:"fear",
3:"happy",
4:"sad",
5:"surprise",
6:"neutral"
}

for emotion in emotions.values():
    os.makedirs(f"data/train/{emotion}", exist_ok=True)
    os.makedirs(f"data/test/{emotion}", exist_ok=True)

for i,row in data.iterrows():

    emotion = emotions[row["emotion"]]

    pixels = np.array(row["pixels"].split(), dtype="uint8")
    image = pixels.reshape(48,48)

    usage = row["Usage"]

    if usage == "Training":
        folder = "train"
    else:
        folder = "test"

    path = f"data/{folder}/{emotion}/{i}.jpg"

    cv2.imwrite(path, image)

print("Dataset converted successfully")
