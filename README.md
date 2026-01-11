🏭 Delhi Winter Air Pollution Hotspot Detection

📌 Overview

Air pollution in Delhi peaks during winter due to meteorological conditions, emissions, and regional factors.
This project uses unsupervised machine learning (DBSCAN) to identify winter air pollution hotspots across Delhi by analyzing multi-pollutant air quality and weather data.

The model detects anomalous high-risk locations (hotspots) that consistently exhibit severe pollution levels, without relying on labeled data.

🎯 Objectives

Identify winter-specific air pollution hotspots in Delhi

Detect anomalous locations with extreme AQI and pollutant levels

Perform cluster-based spatial risk analysis using unsupervised learning

Provide interpretable insights for environmental monitoring and policy analysis

📊 Dataset

Source: Processed Delhi Air Quality Feature Store

Time Span: Multiple years (hourly resolution)

Scope: Delhi city monitoring locations

Key Features Used

Air Pollutants

PM2.5

PM10

NO₂

SO₂

O₃

CO

AQI

Meteorological Factors

Temperature

Humidity

Pressure

Wind Speed

Wind Direction

❄️ Winter Focus

The analysis is restricted to winter months (December–February), when pollution levels are highest due to:

Temperature inversion

Reduced wind dispersion

Increased emissions

🧠 Methodology
1️⃣ Data Preprocessing

Converted timestamps to datetime

Filtered winter months

Aggregated pollutant and weather values by monitoring location

Handled missing values using robust statistics

2️⃣ Feature Scaling

Used RobustScaler to reduce sensitivity to extreme pollution values

3️⃣ Clustering (DBSCAN)

Algorithm: DBSCAN

Why DBSCAN?

No need to predefine number of clusters

Effectively detects outliers as pollution hotspots

Works well with irregular spatial density

Hotspots = DBSCAN noise points (cluster = -1)

4️⃣ Dimensionality Reduction

PCA (Principal Component Analysis) for 2D visualization

Helps interpret cluster separation and anomaly behavior

📈 Visualizations

PM2.5 vs AQI scatter plot with hotspots highlighted

PCA 2D cluster visualization

AQI distribution across clusters

Pollutant correlation heatmap

Hotspots are visually emphasized using red markers.

🏆 Key Results

Successfully detected high-risk pollution hotspots during winter

Identified locations with consistently extreme AQI levels

Provided interpretable cluster-based pollution patterns

💾 Outputs

DELHI_WINTER_CLUSTERS.csv — Cluster assignments for all locations

DELHI_WINTER_HOTSPOTS.csv — Identified pollution hotspots

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

🚀 How to Run
pip install pandas scikit-learn matplotlib seaborn
python Delhi_Winter_Air_Pollution_Hotspot_Detection.py

📌 Applications

Urban air quality monitoring

Environmental risk assessment

Smart city planning

Policy and public health analysis

👤 Author

Mallika Bhardwaj |
Data Science | Machine Learning
