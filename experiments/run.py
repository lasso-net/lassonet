import pickle

import torch
from data_utils import load_dataset
from sklearn.model_selection import train_test_split

from lassonet import LassoNetClassifier
from lassonet.utils import eval_on_path

seed = None
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256
K = 50  # Number of features to select
n_epochs = 1000
dataset = "ISOLET"

# Load dataset and split the data
(X_train_valid, y_train_valid), (X_test, y_test) = load_dataset(dataset)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_valid, y_train_valid, test_size=0.125, random_state=seed
)

# Set the dimensions of the hidden layers
data_dim = X_test.shape[1]
hidden_dim = (data_dim // 3,)

# Initialize the LassoNetClassifier model and compute the path
lasso_model = LassoNetClassifier(
    M=10,
    hidden_dims=hidden_dim,
    verbose=1,
    torch_seed=seed,
    random_state=seed,
    device=device,
    n_iters=n_epochs,
    batch_size=batch_size,
)
path = lasso_model.path(X_train, y_train, X_val=X_val, y_val=y_val)

# Select the features
desired_save = next(save for save in path if save.selected.sum().item() <= K)
SELECTED_FEATURES = desired_save.selected
print("Number of selected features:", SELECTED_FEATURES.sum().item())

# Select the features from the training, validation, and test data
X_train_selected = X_train[:, SELECTED_FEATURES]
X_val_selected = X_val[:, SELECTED_FEATURES]
X_test_selected = X_test[:, SELECTED_FEATURES]

# Initialize another LassoNetClassifier for retraining with the selected features
lasso_sparse = LassoNetClassifier(
    M=10,
    hidden_dims=hidden_dim,
    verbose=1,
    torch_seed=seed,
    random_state=seed,
    device=device,
    n_iters=n_epochs,
)
path_sparse = lasso_sparse.path(
    X_train_selected,
    y_train,
    X_val=X_val_selected,
    y_val=y_val,
    lambda_seq=[0],
    return_state_dicts=True,
)[:1]

# Evaluate the model on the test data
score = eval_on_path(lasso_sparse, path_sparse, X_test_selected, y_test)
print("Test accuracy:", score)

# Save the path
with open(f"{dataset}_path.pkl", "wb") as f:
    pickle.dump(path_sparse, f)
