import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle

if __name__ == '__main__':
    #what is ST
    #ST depression occurs when the J point is displaced below baseline. Just like ST elevation
    #not all ST depression represents myocardial ischemia or an emergent condition

    #column selections
    categorical = ["cp", "restecg", "slope", "ca", "thal"]
    numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    dataset = pd.read_csv("dataset/heart.csv")
    target = dataset["target"]
    target = np.array(target)
    dataset.drop(columns = ["target"] , inplace = True )
    print(dataset.isnull().count())

    # preprocessing before feeding the data
    encoder = OneHotEncoder(dtype = np.int , sparse = False)
    scaller = MinMaxScaler(feature_range=(0, 1))

    encoder.fit(dataset[categorical])
    scaller.fit(dataset[numerical])

    #saving the scaller and one hot encodings so after we can use it
    encoder_file = "preproccesing/encoder.save"
    scaller_file = "preproccesing/scaller.save"

    pickle.dump(encoder, open(encoder_file, 'wb'))
    pickle.dump(scaller, open(scaller_file, 'wb'))

    #scalling and one hot encodings
    data = encoder.fit_transform(dataset[categorical])
    scalled = scaller.fit_transform(dataset[numerical])
    final_data = np.concatenate((data, scalled), axis = 1)

    np.save("preproccesing/data" , final_data)
    np.save("preproccesing/target" , target)
    dataset.to_csv("preproced.csv")

