#importing the neccessary imports
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

class HeartBase(BaseModel):
    #Patients age
    age : float
    # chest pain type : give options of 4 (low , moderate , high , very high)
    cp : int
    #resting blood pressure type often ==> value between (94 , 200)
    trestbps : float
    #serum cholestoral in mg/dl
    chol : float
    #resting electrocardiographic results ==> (values 0,1,2)
    restecg : int
    #maximum heart rate achieved ==> any float value
    thalach : float
    #ST depression induced by exercise relative to rest ==> float (0 , max to max 7)
    oldpeak : float
    #the slope of the peak exercise ST segment ==> optins (0, 1, 2)
    slope : int
    #number of major vessels (0-3) colored by flourosopy ==> (0, 1, 2, 3)
    ca : int
    #thal ==> value between ==> (0, 1, 2, 3)
    thal : int

#Creating the FastAPI Object
app = FastAPI()

#FIle path
model_path = "trained_model/random.model"

#loading the models for the given task Logistic Model with 84 % accuracy
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
with open("preproccesing/encoder.save", "rb") as enc:
    encoder = pickle.load(enc)
with open("preproccesing/scaller.save", "rb") as sclr:
    scaller = pickle.load(sclr)

@app.get("/")
def read_root():
    return {"Hello from pratik korat"}

@app.get("/info")
def give_info():
    return {"welcome to medihub with ML AICTE_MODEL"}

@app.post("/predict")
def read_item(data : HeartBase):
    data = data.dict()
    cat_value = []
    num_value = []

    cat_value.append([data["cp"] , data["restecg"] , data["slope"] , data["ca"] , data["thal"]])

    num_value.append([data["age"] , data["trestbps"], data["chol"], data["thalach"], data["oldpeak"]])

    scalled = scaller.transform(num_value)
    categorized = encoder.transform(cat_value)
    #
    final_data = np.concatenate((scalled, categorized), axis=1)

    pred = model.predict(final_data)
    print(pred)
    if pred[0] > 0.5:
        prediction = "you should consult higher doctor"
    if pred[0] < 0.5:
        prediction = "you are healthy"
    return {
        "prediction": prediction
    }

if __name__ == '__main__':
    pass