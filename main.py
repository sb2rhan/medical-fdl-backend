from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from datatypes import IrisFeatures

app = FastAPI(title="Medical FDL API", description="An API to get model predictions", version="1.0.0")

@app.get("/")
def home():
    return {"message": "Welcome to the Medical FDL API"}

# Load Iris dataset
iris = load_iris()

# Extract features and labels
X, y = iris.data, iris.target

# Train the model
clf = GaussianNB()
clf.fit(X, y)

@app.post("/predict")
def predict(data: IrisFeatures):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return {"class": iris.target_names[class_idx]}