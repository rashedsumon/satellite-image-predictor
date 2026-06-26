# EuroSAT Land Cover Classification App

This AI-powered web application automatically classifies satellite images into 10 distinct land cover categories using a Deep Learning model built in PyTorch and deployed via Streamlit Cloud.

---

## Machine Learning Concepts

### 1. User Input (The Features / Independent Variables)
In machine learning, the user input (often called $X$) is the data you feed into the model so it can make a prediction.
* **In this EuroSAT example:** The user input is a single $64 \times 64$ pixel satellite image covering a specific patch of land.
* **What it looks like to the computer:** A grid of numbers (a matrix or tensor) representing the color intensity or spectral bands (Red, Green, Blue) captured by the satellite.

### 2. Model Output (The Target / Prediction)
The model output (often called $\hat{y}$) is what the trained algorithm predicts or decides after analyzing the input.
* **In this EuroSAT example:** The model output is a specific land cover category label along with a confidence percentage.

---

## The 10 Possible Land Cover Outputs
The model categorizes any input image into one of these 10 distinct labels:
1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. Permanent Crop
8. Residential
9. River
10. Sea/Lake