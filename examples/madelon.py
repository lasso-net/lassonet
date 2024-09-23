# %%
import openml

dataset = openml.datasets.get_dataset(1485)
X, *_ = dataset.get_data(dataset_format="dataframe")
# %%
# class is last column
X, y = X.iloc[:, :-1], X.iloc[:, -1]

# %%
X = X.to_numpy() / 500
# %%
y = (y == "2").to_numpy()
# %%

from lassonet import LassoNetClassifier

# %%
model = LassoNetClassifier(hidden_dims=(10, 10))

oracle, order, wrong, paths, prob = model.stability_selection(X, y, n_models=500)

# %%
sorted(model.feature_importances_)
# %%
