from pydantic import BaseModel


class HeartBase(BaseModel):
    #Patients age
    age : float

    # chest pain type : give options of 4 (low , moderate , high , very high)
    cp : str

    #resting blood pressure type often ==> value between (94 , 200)
    trestbps : float

    #serum cholestoral in mg/dl
    chol : float

    #resting electrocardiographic results ==> (values 0,1,2)
    restecg : float

    #maximum heart rate achieved ==> any float value
    thalach : float

    #ST depression induced by exercise relative to rest ==> float (0 , max to max 7)
    oldpeak : float

    #the slope of the peak exercise ST segment ==> optins (0, 1, 2)
    slope : float

    #number of major vessels (0-3) colored by flourosopy ==> (0, 1, 2, 3)
    ca : float

    #thal ==> value between ==> (0, 1, 2, 3)
    thal : float