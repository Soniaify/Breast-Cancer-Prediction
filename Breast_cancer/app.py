from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

app = Flask(__name__)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = pd.DataFrame(data.target, columns=["target"])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train models
log_model = LogisticRegression(max_iter=5000)
log_model.fit(x_train, y_train.values.ravel())

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train.values.ravel())

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        try:
            # Get all 30 features from form
            features = [float(request.form.get(f)) for f in data.feature_names]
            features_df = pd.DataFrame([features], columns=data.feature_names)

            # Select and use the chosen model
            if model_choice == "logistic":
                result = log_model.predict(features_df)[0]
            else:
                result = knn_model.predict(features_df)[0]

            # 0 = malignant, 1 = benign in sklearn's dataset
            prediction = "Malignant" if result == 0 else "Benign"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", feature_names=data.feature_names, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
