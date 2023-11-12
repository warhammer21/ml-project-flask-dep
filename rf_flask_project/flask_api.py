import pandas as pd
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pickle
import pandas as pd
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


# loading the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

# Build the model
model = RandomForestClassifier(n_estimators=10)
# Train the classifier
model.fit(X_train, y_train)



# path = 'rf.pkl'
# with open('rf.pkl','rb') as model_file:
#     model = pickle.load(model_file)
app = Flask(__name__)
swagger = Swagger(app)
@app.route("/predict")
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    prediction = model.predict(np.array([[s_length,
                                          s_width,
                                          p_length,
                                          p_width]]))
    return str(prediction)
@app.route("/predict_file",methods = ["POST"])
def predict_iris_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get('input_file'),header = None)
    prediction = model.predict(input_data)
    return str(list(prediction))
if __name__ == '__main__':
    app.run()
    print('yes it hit here')

