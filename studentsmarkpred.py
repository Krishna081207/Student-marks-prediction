# Task 1 (Simple Version): Predicting Student Marks using Linear Regression
# Libraries: pandas, scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Simple dataset (Hours studied, Sleep hours, Extra activity, Marks)
data = {
    "Hours_Studied":   [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Sleep_Time":      [9, 8, 8, 7, 7, 6, 6, 6, 5],
    "Extra_Activity":  [5, 4, 3, 3, 2, 2, 1, 1, 1],
    "Marks":           [50, 55, 60, 65, 72, 78, 85, 90, 95]
    " Attendence %":   [70 ,60 ,60 ,55 ,80 ,75 ,63 ,95 ,90]
}
df = pd.DataFrame(data)

# Features and Target
X = df[["Hours_Studied", "Sleep_Time", "Extra_Activity","Attendence"]]
y = df["Marks"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict example
sample_student = [[6, 7, 2, 85]]  # 6 hrs study, 7 hrs sleep, 2 hrs activities
pred = model.predict(sample_student)[0]

print("Predicted Marks:", round(pred, 2))
