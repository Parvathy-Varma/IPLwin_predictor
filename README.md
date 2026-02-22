# 🏏 IPL Win Predictor

A Machine Learning based web application that predicts the **winning probability of an IPL team during the second innings** using real match data.

Built using:

* Python 🐍
* Pandas & NumPy
* Scikit-Learn
* Streamlit
* Logistic Regression

---

## 📌 Project Overview

This project predicts the **probability of winning** for the batting team in the second innings of an IPL match based on:

* Batting Team
* Bowling Team
* Host City
* Target Score
* Current Score
* Overs Completed
* Wickets Fallen

The model calculates:

* Runs Left
* Balls Left
* Current Run Rate (CRR)
* Required Run Rate (RRR)
* Wickets Remaining

Then it outputs the winning probability for both teams.

---

## 📂 Dataset Used

The project uses IPL historical data:

* `matches.csv`
* `deliveries.csv`

These datasets contain ball-by-ball match information and match-level details.

---

## ⚙️ Machine Learning Model

Algorithm used:

👉 **Logistic Regression**
Why?

* It is a **binary classification model**
* It provides **probability output**
* Suitable for predicting win/lose scenarios

Pipeline includes:

* OneHotEncoder (for categorical features)
* ColumnTransformer
* Logistic Regression (liblinear solver)

Model is saved as:

```
pipe.pkl
```

---

## 🧠 Features Used for Prediction

| Feature      | Description          |
| ------------ | -------------------- |
| batting_team | Team chasing         |
| bowling_team | Opponent team        |
| city         | Match venue          |
| runs_left    | Runs required to win |
| balls_left   | Balls remaining      |
| wickets      | Wickets remaining    |
| total_runs_x | Target score         |
| crr          | Current run rate     |
| rrr          | Required run rate    |

---

## 🖥️ Web Application

The frontend is built using **Streamlit**.

Main file:

```
app.py
```

User selects:

* Batting team
* Bowling team
* City
* Target
* Current score
* Overs completed
* Wickets fallen

Click **Predict Probability** to see win percentages.

---

## 🚀 How to Run This Project

### 1️⃣ Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

### 2️⃣ Run the Application

```bash
streamlit run app.py
```

The app will open in your browser.

---

## 📊 Model Training Script

The model training is done in:

```
ipl.py
```

Steps included:

* Data Cleaning
* Removing DLS matches
* Feature Engineering
* Creating runs_left, balls_left, wickets
* Train-test split
* Pipeline creation
* Model training
* Saving model as pipe.pkl

---

## 📁 Project Structure

```
IPL-Win-Predictor/
│
├── app.py
├── ipl.py
├── pipe.pkl
├── matches.csv
├── deliveries.csv
└── README.md
```

---

## 🎯 Future Improvements

* Add Over-by-Over win probability graph
* Deploy using Streamlit Cloud
* Improve accuracy using Random Forest / XGBoost
* Add more recent IPL seasons
* Improve UI design

---

## 📈 Sample Output

Example:

```
Chennai Super Kings – 64%
Mumbai Indians – 36%
```

---

## 🏆 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Logistic Regression

---

## 👩‍💻 Author

Developed as a Machine Learning project to understand:

* Feature Engineering
* Classification
* Model Pipelines
* Probability Prediction
* Web App Deployment
