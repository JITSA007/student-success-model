import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load data and separate clues (X) from answer (y)
data = pd.read_csv("student_exam_performance.csv")
target_column = "pass_status"
columns_to_drop = ["student_id", "exam_score", "performance_grade", "performance_level"]

X = data.drop(columns=columns_to_drop + [target_column])
y = data[target_column]

# 2. Convert text columns to numbers
X_numeric = pd.get_dummies(X, drop_first=True)
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y_numeric, test_size=0.2, random_state=42
)

# 4. Train the model
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Draw and save the feature importance chart
print("Generating Feature Importance Chart...")
xgb.plot_importance(model, max_num_features=10, title="Top 10 Drivers of Student Success")
plt.tight_layout()
plt.savefig("feature_importance.png")

print("Chart successfully saved as 'feature_importance.png'!")