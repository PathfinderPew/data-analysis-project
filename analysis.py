import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')

# ── 1. Load data ──────────────────────────────────────────────────
df = pd.read_csv('data/train.csv')
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nIrrigation Need distribution:\n", df['Irrigation_Need'].value_counts())

# ── 2. Clean & transform ──────────────────────────────────────────
# No major nulls expected, but let's be safe
df = df.dropna()

# Create a numeric version of the target for correlation
irrigation_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Irrigation_Score'] = df['Irrigation_Need'].map(irrigation_map)

# Bin temperature into ranges
df['Temp_Group'] = pd.cut(df['Temperature_C'],
    bins=[0, 15, 25, 35, 60],
    labels=['Cool', 'Mild', 'Warm', 'Hot'])

# ── 3. Key insights ───────────────────────────────────────────────
print("\nAvg Rainfall by Irrigation Need:")
print(df.groupby('Irrigation_Need')['Rainfall_mm'].mean().round(2))

print("\nAvg Soil Moisture by Irrigation Need:")
print(df.groupby('Irrigation_Need')['Soil_Moisture'].mean().round(2))

print("\nIrrigation Need by Region:")
print(pd.crosstab(df['Region'], df['Irrigation_Need'], normalize='index').round(2))

# ── 4. Visualizations ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Irrigation Need — Exploratory Data Analysis', fontsize=15)

# Chart 1: Irrigation Need counts
order = ['Low', 'Medium', 'High']
df['Irrigation_Need'].value_counts().reindex(order).plot(
    kind='bar', ax=axes[0,0],
    color=['#3B8BD4', '#1D9E75', '#EF9F27'], edgecolor='white')
axes[0,0].set_title('Irrigation Need distribution')
axes[0,0].set_xlabel('')
axes[0,0].tick_params(axis='x', rotation=0)

# Chart 2: Rainfall vs Irrigation Need
df.boxplot(column='Rainfall_mm', by='Irrigation_Need',
           ax=axes[0,1], positions=[0,1,2])
axes[0,1].set_title('Rainfall by Irrigation Need')
axes[0,1].set_xlabel('Irrigation Need')
axes[0,1].set_xticks([0,1,2])
axes[0,1].set_xticklabels(['Low','Medium','High'])
plt.sca(axes[0,1])
plt.title('Rainfall by Irrigation Need')

# Chart 3: Avg Irrigation Score by Soil Type
soil_avg = df.groupby('Soil_Type')['Irrigation_Score'].mean().sort_values()
soil_avg.plot(kind='barh', ax=axes[1,0], color='#534AB7', edgecolor='white')
axes[1,0].set_title('Avg irrigation need by soil type')
axes[1,0].set_xlabel('Irrigation Score (0=Low, 2=High)')

# Chart 4: Correlation heatmap (numeric columns only)
num_cols = ['Soil_pH','Soil_Moisture','Temperature_C','Humidity',
            'Rainfall_mm','Sunlight_Hours','Wind_Speed_kmh',
            'Field_Area_hectare','Previous_Irrigation_mm','Irrigation_Score']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', ax=axes[1,1],
            cmap='coolwarm', annot_kws={'size': 7})
axes[1,1].set_title('Correlation matrix')
plt.sca(axes[1,1])
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)

plt.tight_layout()
plt.savefig('data/analysis_results.png', dpi=150, bbox_inches='tight')
print("\nDone! Chart saved to data/analysis_results.png")