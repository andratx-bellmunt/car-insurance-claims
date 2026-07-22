# 🚗 Car Insurance Claims Prediction

Two Machine Learning models -Poisson GLM and XGBoost- are used to predict number of claims for car insurance data.


## 🔗 Live Reports
The full interactive data storytelling reports—complete with narrative analysis and dynamic visualizations—are hosted live on my portfolio:

👉 **[View the Interactive Portfolio Report (EDA)](https://andratx-bellmunt.github.io/portfolio/projects/carinsurance/eda.html)**

👉 **[View the Interactive Portfolio Report (Predictive Modeling)](https://andratx-bellmunt.github.io/portfolio/projects/carinsurance/modeling.html)**


## 🛠️ Project Architecture

This repository is engineered using modular, industry-standard Python data pipelines. Heavy plotting logic and helper functions are isolated from the core narrative to keep code execution clean and readable.

```text
.
├── data/                  # Source CSV datasets
├── docs/                  # Clean production artifacts (rendered HTML)
├── src/
│   ├── notebooks/         # Narrative-driven analysis and modeling notebooks
│   └── utils/             # Modular Python utility scripts (e.g., plots.py)
├── templates/             # HTML injection fragments (analytics & tracking tokens)
├── _quarto.yml            # Automated Quarto build configurations
├── pyproject.toml         # Fast, reproducible dependency configurations (via uv)
└── uv.lock                # Deterministic lockfile for exact environment state
```


## 🚀 Quick Start & Reproducibility

This project utilizes `uv` for Python dependency management. Follow these steps to clone the repo and run the environment locally:

### Clone the Repository

```bash
git clone https://github.com/andratx-bellmunt/car-insurance-claims.git
cd car-insurance-claims
```

### Set Up the Environment

Ensure you have `uv` installed, then synchronize the environment (this will automatically create a virtual environment and install all pinned versions):
```bash
uv sync
```

### Explore the Code

To run the notebooks with your active `uv` environment:

```bash
uv run jupyter notebook src/notebooks/eda.ipynb
uv run jupyter notebook src/notebooks/modeling.ipynb
```

## 📊 Key Findings

* **Performance:** The **XGBoost model** is the clear winner, explaining **9.17%** of the deviance on unseen data, clearly improving the predictive power of the Possion GLM model (**5.26%**).

* **Segmentation:** Lorenz curves confirm that XGBoost is significantly more capable of isolating high-risk segments, which allows for more competitive pricing and better risk selection.

* **Model Reliability:** While slight overfitting was observed in the XGBoost model, it remains robust. The "top errors" analysis reveals that remaining inaccuracies are due to random claim volatility (3-claim outliers) rather than model failure, affecting all models equally.


## 🗃️ Data Sources

We use the French Motor Claims Dataset, which is [publicly available](https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq).


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

