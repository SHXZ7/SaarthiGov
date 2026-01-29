import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib

INTENTS = ["documents","eligibility","process","timeline","fees","correction"]
SERVICES = ["ration_card","birth_certificate","unemployment"]

# Load data
with open("training_data.json") as f:
    data = json.load(f)

X = []
y = []

for row in data:
    service_vec = [1 if row["service"] == s else 0 for s in SERVICES]
    intent_vec = [1 if row["current_intent"] == i else 0 for i in INTENTS]
    X.append(service_vec + intent_vec)
    y.append(row["next_intents"])

X = np.array(X)

mlb = MultiLabelBinarizer(classes=INTENTS)
y = mlb.fit_transform(y)

model = OneVsRestClassifier(LogisticRegression())
model.fit(X, y)

joblib.dump(model, "next_step_model.pkl")
joblib.dump(mlb, "label_binarizer.pkl")
