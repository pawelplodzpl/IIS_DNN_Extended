import keras
from keras.models import load_model
from keras.models import Model, Sequential
from load_data import cherryCar_loadData_full



model = load_model('model/model_CherryCar.h5')

x_train, y_train, _, _ = cherryCar_loadData_full()

results = model.predict(x_train)

for i, angle in enumerate(results):
    print("is {:10.10f}, should be {:10.10f}, diff {:10.10f}".format(float(results[i]), y_train[i], float(results[i]) - y_train[i]))




