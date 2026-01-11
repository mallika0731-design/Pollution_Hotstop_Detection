import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load YOUR dataset
df = pd.read_csv('delhi_air_quality_processed.csv')
print(f"✅ Dataset: {df.shape[0]:,} rows")

# Filter winter months (Dec, Jan, Feb)
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
winter_df = df[df['event_timestamp'].dt.month.isin([12, 1, 2])]
print(f"❄️ Winter records: {len(winter_df):,}")

# YOUR EXACT COLUMNS from dataset
features = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi']
weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']

# Aggregate by location
agg_dict = {col: 'mean' for col in features + weather_cols if col in winter_df.columns}
agg_df = winter_df.groupby(['city', 'location_id']).agg(agg_dict).reset_index()
print(f"📍 {len(agg_df)} unique Delhi locations")

# Use pollution features only
pollution_features = [f for f in features if f in agg_df.columns]
print(f"🔥 Using: {pollution_features}")

# Scale data
X = agg_df[pollution_features].fillna(agg_df[pollution_features].median())
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=1.2, min_samples=2)
clusters = dbscan.fit_predict(X_scaled)
agg_df['cluster'] = clusters

# Hotspots = noise points (-1)
hotspots = agg_df[agg_df['cluster'] == -1].sort_values('aqi', ascending=False)

# RESULTS
print("\n" + "="*60)
print("🚨 DELHI WINTER POLLUTION HOTSPOTS")
print("="*60)
if len(hotspots) > 0:
    print(hotspots[['city', 'aqi', 'pm25', 'pm10']].round(2).head(10))
else:
    print("No outliers detected - all locations form dense clusters")

print(f"\n📊 SUMMARY:")
print(f"Total locations: {len(agg_df)}")
print(f"Hotspots found: {len(hotspots)} ({len(hotspots)/len(agg_df)*100:.1f}%)")

# VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. PM25 vs AQI (Hotspots = red stars)
colors = ['green' if c >= 0 else 'red' for c in clusters]
axes[0,0].scatter(agg_df['pm25'], agg_df['aqi'], c=colors, s=100, alpha=0.7)
if len(hotspots) > 0:
    axes[0,0].scatter(hotspots['pm25'], hotspots['aqi'], 
                     c='darkred', s=250, marker='*', label='Hotspots')
axes[0,0].set_xlabel('PM2.5'); axes[0,0].set_ylabel('AQI')
axes[0,0].set_title('DBSCAN Hotspots (Red Stars)')
axes[0,0].legend()

# 2. PCA 2D view
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
colors_pca = ['green' if c >= 0 else 'red' for c in clusters]
axes[0,1].scatter(X_pca[:,0], X_pca[:,1], c=colors_pca, s=100)
if len(hotspots) > 0:
    axes[0,1].scatter(X_pca[hotspots.index,0], X_pca[hotspots.index,1], 
                     c='darkred', s=250, marker='*')
axes[0,1].set_title(f'PCA View ({pca.explained_variance_ratio_.sum():.1%} variance)')
axes[0,1].set_xlabel('PC1'); axes[0,1].set_ylabel('PC2')

# 3. AQI distribution by cluster
sns.boxplot(data=agg_df, x='cluster', y='aqi', ax=axes[1,0])
axes[1,0].set_title('AQI by Cluster')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Feature correlation heatmap
corr_matrix = agg_df[pollution_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
axes[1,1].set_title('Pollutant Correlations')

plt.tight_layout()
plt.show()

# Export results
agg_df.to_csv('DELHI_WINTER_CLUSTERS.csv', index=False)
if len(hotspots) > 0:
    hotspots.to_csv('DELHI_WINTER_HOTSPOTS.csv', index=False)
    print(f"\n💾 SAVED:")
    print(f"  → DELHI_WINTER_HOTSPOTS.csv ({len(hotspots)} hotspots)")
print(f"  → DELHI_WINTER_CLUSTERS.csv ({len(agg_df)} locations)")
print("✅ COMPLETE! Red stars = your pollution hotspots!")
