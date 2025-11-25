import pandas as pd

data = pd.read_csv("all_data.csv")

# write to summary file
summary_label = data.label.value_counts().reset_index()
summary_label.to_csv("label_distribution.csv", index=False)

summary_category = data.category.value_counts().reset_index()
summary_category.to_csv("category_distribution.csv", index=False)
