from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report , accuracy_score
import pickle
import numpy as np

def read_dataset():
    data = np.load("preproccesing/data.npy")
    label = np.load("preproccesing/target.npy")
    print("data loaded for the training")
    print(data.shape , label.shape)
    np.random.seed(41)

    train_x , test_x , train_y , test_y = \
        train_test_split(data,
                         label,
                         random_state = np.random.randint(1 , 100),
                         shuffle = True)
    return {
        "train_x" : train_x,
        "train_y" : train_y,
        "test_x" : test_x,
        "test_y" : test_y
    }

def enlisted_model(max_iter:int):
    #list of probable models
    models = {
        "svc" : SVC(kernel="poly",
                    degree=3,
                    gamma = "auto" ,
                    max_iter = max_iter if max_iter else 1000),
        "logistic" : LogisticRegression(penalty = "l2",
                                        max_iter = max_iter if max_iter else 1000),

        "sgd" :  SGDClassifier(max_iter = max_iter if max_iter else 1000),

        "random" : RandomForestClassifier(),

        "adaboost" : AdaBoostClassifier()
    }

    model_list = models.keys()

    dataset = read_dataset()

    for i in model_list:
        print(f"{i} mdoel trained...")

        model = models[i]
        model.fit(dataset["train_x"] , dataset["train_y"])

        output = model.predict(dataset["test_x"])

        acc = accuracy_score(dataset["test_y"], output)
        report = classification_report(dataset["test_y"] , output)
        print(f"{i} accuracy : {acc}")
        with open(f"report/{i}.txt" , "w") as f:
            f.write(f"accuracy: {acc * 100}%\n")
            f.write(f"report : \n{report}")

        pickle.dump(model, open(f"trained_model/{i}.model", 'wb'))

if __name__ == "__main__":
    enlisted_model(5000)