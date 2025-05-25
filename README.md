# 💎 Diamond Price Prediction

This project aims to predict diamond prices based on various attributes using machine learning techniques. It encompasses data preprocessing, model training, evaluation, and deployment through a Flask web application.

---

## 📁 Project Structure

```
DiamondPricePrediction/
├── artifacts/
├── mlartifacts/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   └── model_evaluation.py
├── templates/
│   └── index.html
├── application.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📊 Dataset

The dataset used for this project contains information on various diamonds, including features like carat weight, cut, color, clarity, depth, table dimensions, and price.

* **Source**: [Kaggle - Diamonds Dataset](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

* **Features**:

  * `carat`: Weight of the diamond
  * `cut`: Quality of the cut (e.g., Fair, Good, Very Good, Premium, Ideal)
  * `color`: Diamond color grading (from D to J)
  * `clarity`: Measure of diamond clarity (e.g., I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
  * `depth`: Total depth percentage
  * `table`: Width of the top of the diamond relative to the widest point
  * `x`: Length in mm
  * `y`: Width in mm
  * `z`: Depth in mm
  * `price`: Price in USD (target variable)

---

## 🧠 Machine Learning Models

The project explores multiple regression algorithms to predict diamond prices:

* Linear Regression
* Lasso Regression
* Ridge Regression
* Elastic Net Regression
* Decision Tree Regressor
* Random Forest Regressor
* K-Nearest Neighbors Regressor

Each model is trained and evaluated to determine the best-performing algorithm based on metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

---

## 🛠️ Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kishan42/DiamondPricePrediction.git
   cd DiamondPricePrediction
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   python application.py
   ```

   The Flask app will start, and you can access it by navigating to `http://localhost:5000` in your web browser.

---

## 📈 Exploratory Data Analysis (EDA)

The `notebooks/eda.ipynb` notebook provides an in-depth analysis of the dataset, including:

* Data distribution and visualization
* Correlation analysis
* Outlier detection
* Feature importance

This analysis aids in understanding the data and informs feature selection and engineering decisions.

---

## 🧪 Model Training & Evaluation

The `src/` directory contains modular scripts for:

* **Data Ingestion** (`data_ingestion.py`): Loading and splitting the dataset.
* **Data Transformation** (`data_transformation.py`): Handling missing values, encoding categorical variables, and feature scaling.
* **Model Training** (`model_trainer.py`): Training various regression models.
* **Model Evaluation** (`model_evaluation.py`): Evaluating model performance using appropriate metrics.

The best-performing model is saved for deployment.

---

## 🌐 Web Application

The Flask web application (`application.py`) provides a user-friendly interface to input diamond features and obtain price predictions. The `templates/index.html` file defines the front-end structure using HTML.

---

## 📌 Future Enhancements

* Implement hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Deploy the application on cloud platforms like AWS or Heroku.
* Integrate user authentication for personalized experiences.
* Enhance the front-end with responsive design and better UX/UI.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
