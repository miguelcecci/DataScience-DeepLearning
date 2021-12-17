from sys import argv
import joblib
import importlib
import pandas as pd

loader = importlib.machinery.SourceFileLoader('bk.py', '../bk.py')

bk = loader.load_module()

_, filename, destination = argv

def run_prediction():
    data = pd.read_csv(filename, sep=';')
    model = joblib.load(bk.MODEL_FILE_PATH)
    data = bk.treat_data(data)
    preds = model.predict(data)
    preds = pd.DataFrame({'predictions': preds})
    preds.to_csv(destination, index=False)
    print("Modelo executado e predicoes salvas em: {}".format(destination))

run_prediction()
