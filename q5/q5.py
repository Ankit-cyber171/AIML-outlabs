import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

# -------------------------------
# Load data
# -------------------------------
def load_data():
    train = np.genfromtxt('q5_train.csv', delimiter=',', skip_header=1)
    test  = np.genfromtxt('q5_test.csv', delimiter=',', skip_header=1)
    return train, test


def plot_correlation_heatmap(X, y):
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y

    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap (Features & Target)")
    plt.show()


# -------------------------------
# NEW FUNCTION 2:
# Scatter plots (feature vs y)
# -------------------------------
def plot_feature_scatter(X, y):
    num_features = X.shape[1]
    feature_names = [f"Feature_{i+1}" for i in range(num_features)]

    plt.figure(figsize=(12, 4 * num_features))

    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.scatter(X[:, i], y, alpha=0.6)
        plt.xlabel(feature_names[i])
        plt.ylabel("y")
        plt.title(f"{feature_names[i]} vs y")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    train, test = load_data()

    X_train = train[:, :-1]
    y_train = train[:, -1]

    lambdas = np.logspace(-4,3,1000)
    log_lambdas = np.log10(lambdas)

    ridge_coefs = []
    lasso_coefs = []

    # -------------------------------
    # Train models for each lambda
    # -------------------------------
    for lam in lambdas:
        # Ridge
        ridge_pipe = Pipeline([
            ("ridge", Ridge(alpha=lam))
        ])
        ridge_pipe.fit(X_train, y_train)
        ridge_coefs.append(ridge_pipe.named_steps["ridge"].coef_)

        # Lasso
        lasso_pipe = Pipeline([
            ("scaler",StandardScaler()),
            ("lasso", Lasso(alpha=lam, max_iter=10000))
        ])
        lasso_pipe.fit(X_train, y_train)
        lasso_coefs.append(lasso_pipe.named_steps["lasso"].coef_)

    ridge_coefs = np.array(ridge_coefs)  # (len(lambdas), d)
    lasso_coefs = np.array(lasso_coefs)

    # -------------------------------
    # Plot Ridge coefficient path
    # -------------------------------
    plt.figure(figsize=(8, 5))
    for i in range(ridge_coefs.shape[1]):
        plt.plot(log_lambdas, ridge_coefs[:, i], marker='o', label=f'Feature {i}')

    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient value")
    plt.title("Ridge Regression Coefficient Path")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------
    # Plot Lasso coefficient path
    # -------------------------------
    plt.figure(figsize=(8, 5))
    for i in range(lasso_coefs.shape[1]):
        plt.plot(log_lambdas, lasso_coefs[:, i], marker='o', label=f'Feature {i}')

    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient value")
    plt.title("Lasso Regression Coefficient Path")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plot_correlation_heatmap(X_train, y_train)
    plot_feature_scatter(X_train, y_train)