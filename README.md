# Crop Prediction System

**Overview**

_Agriculture plays a vital role in the economy of many countries, and choosing the right crop for cultivation is one of the most important decisions for farmers. Traditionally, this decision depends on experience and intuition, which may not always guarantee the best results._

This project provides a machine learning-based crop prediction system that suggests the most suitable crop to cultivate based on soil and environmental conditions. By leveraging data-driven insights, the system helps in:

1. Maximizing yield
2. Reducing resource wastage
3. Improving decision-making in agriculture
4. The model uses features such as soil nutrients, rainfall, pH value, temperature, and humidity to predict the crop that is most likely to thrive.

**Key Features**

1. Accurate Predictions: Uses machine learning models trained on real-world datasets.
2. User Input Flexibility: Farmers can provide environmental parameters and receive crop recommendations.
3. Data Visualization: Insights on soil quality, weather, and correlations are visualized for better understanding.
4. Deployment Ready: The project includes code for running a web app using Flask or Streamlit.
5. Scalable: Can be extended with more data (e.g., soil moisture, sunlight, fertilizer use).

**Input Parameters**

The model works on the following key features:

1. N: Nitrogen content in soil
2. P: Phosphorus content in soil
3. K: Potassium content in soil
4. pH: Acidity/alkalinity of soil
5. Rainfall: Annual rainfall (in mm)
6. Temperature: Average temperature (°C)
7. Humidity: Average relative humidity (%)

**Technologies Used**

1. Programming Language: Python
2. Libraries: Data Handling: Pandas, NumPy
3. Machine Learning: Scikit-learn, XGBoost, LightGBM
4. Visualization: Matplotlib, Seaborn
5. Deployment: Flask / Streamlit
6. Version Control: Git, GitHub

Folder Structure 
├── data/                      # Dataset files (CSV or external links)
├── notebooks/                 # Jupyter notebooks for EDA and model testing
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── src/                       # Source code
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── train_model.py         # Model training script
│   ├── evaluate_model.py      # Model evaluation script
│   └── predict.py             # Prediction script
├── app/                       # Web application code
│   ├── app.py                 # Flask/Streamlit app
│   └── templates/             # HTML templates (if Flask)
├── requirements.txt           # List of dependencies
└── README.md                  # Documentation

**Installation and Usage**
**1. Clone the Repository**
git clone https://github.com/your-username/crop-prediction.git
cd crop-prediction

**2. Install Dependencies**
pip install -r requirements.txt

**3. Train the Model**
python src/train_model.py

**4. Make Predictions**

1. Option A: Using the script:- 
python src/predict.py --N 90 --P 42 --K 43 --temperature 23 --humidity 82 --ph 6.5 --rainfall 200

2. Option B: Run the web app
3. streamlit run app/app.py

**Dataset**

1. The dataset used for training includes environmental and soil conditions mapped to crops.
2. If you are using a public dataset, you should provide its source here (e.g., Kaggle Crop Recommendation Dataset).
3. Model Training and Evaluation
4. The system was trained using multiple machine learning models such as:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

_The best-performing model was selected based on accuracy, precision, recall, and F1-score. Cross-validation techniques were applied to avoid overfitting._

**Example Evaluation Metrics**

- Accuracy: 95%
- Precision: 94%
- Recall: 93%
- F1 Score: 93%

(These numbers are placeholders; update with actual results from your experiments.)

5. Results and Visualizations

**The project includes visualizations such as:**

- Distribution of soil nutrients
- Correlation between parameters
- Crop frequency per region
- Confusion matrix for model performance
- Future Improvements
- Integrate IoT sensors for real-time soil and weather data collection.
- Add more features like soil moisture, sunlight duration, and fertilizer use.
- Experiment with deep learning models (e.g., LSTMs for time-series weather data).
- Deploy as a mobile-friendly app for direct farmer usage.
- Provide multilingual support for better accessibility.
- Contribution Guidelines
- Contributions are welcome. To contribute:
- Fork the repository.

Create a new branch (feature-branch).

Commit your changes.

Push the branch and create a Pull Request.


 # License

This project is licensed under the Apache License 2.0 – you are free to use, modify, and distribute this project, provided that you include a copy of the license in any derivative work.

_For more details, please see the LICENSE file or visit Apache License 2.0._
