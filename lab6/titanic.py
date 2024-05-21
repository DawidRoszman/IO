import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("./titanic.csv")

# Delete first column
df = df.drop(df.columns[0], axis=1)

print(df.head(10))

items = set()
for col in df:
    items.update(df[col].unique())

print(items)

itemset = set(items)

encoded_vals = []
for index, row in df.iterrows():
    rowset = set(row)
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]

ohe_df = pd.DataFrame(encoded_vals)

freq_items = apriori(ohe_df, min_support=0.005, use_colnames=True, verbose=1)
print(freq_items.head(7))

rules = association_rules(freq_items, metric="confidence", min_threshold=0.8)
# Filter rules pointing to survivors
rules = rules[rules["consequents"] == {"Yes"}]
rules = rules.sort_values("confidence", ascending=False)
print(rules.head())

plt.scatter(rules["support"], rules["confidence"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("confidence")
plt.title("Support vs Confidence")
plt.show()

plt.scatter(rules["support"], rules["lift"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()

fit = np.polyfit(rules["lift"], rules["confidence"], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules["lift"], rules["confidence"], "yo", rules["lift"], fit_fn(rules["lift"]))
plt.show()
