import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed_data.csv")

# Rating distribution
sns.histplot(df['Rating'], bins=20)
plt.title("Rating Distribution")
plt.show()

# Top genres
df['Genre'].value_counts().head(10).plot(kind='bar')
plt.title("Top Genres")
plt.show()