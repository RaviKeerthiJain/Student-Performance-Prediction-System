import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("../data/student.csv")

print("First 5 Rows")
print(df.head())

# =========================
# Remove Unwanted Column
# =========================

if 'student_id' in df.columns:
    df = df.drop('student_id', axis=1)

# =========================
# Encode Target Column
# =========================

le = LabelEncoder()

df['grade'] = le.fit_transform(df['grade'])

# =========================
# EDA VISUALIZATION
# =========================

sns.scatterplot(
    x=df['attendance_percentage'],
    y=df['total_score']
)

plt.title("Attendance vs Total Score")
plt.show()

# =========================
# Features and Target
# =========================

X = df[
    [
        'weekly_self_study_hours',
        'attendance_percentage',
        'class_participation',
        'total_score'
    ]
]

y = df['grade']

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Logistic Regression Model
# =========================

model = LogisticRegression()

model.fit(X_train, y_train)

# =========================
# Prediction
# =========================

y_pred = model.predict(X_test)

# =========================
# Accuracy
# =========================

acc = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {acc:.2f}%")

# =========================
# Classification Report
# =========================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("../outputs/confusion_matrix.png")

plt.show()

# =========================
# New Student Prediction
# =========================

new_student = [[10, 90, 8, 95]]

prediction = model.predict(new_student)

predicted_grade = le.inverse_transform(prediction)

print("\nPrediction for New Student:")

print("Predicted Grade:", predicted_grade[0])