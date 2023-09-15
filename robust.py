import warnings
warnings.simplefilter("ignore")
import pandas as pd
from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/felicidad_corrupt.csv")
    print(dataset.head())
    
    X = dataset.drop(['country','score','rank'], axis=1)
    y = dataset[['score']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.3, random_state=42)
    
    estimadores = {
        # C, podemos controlar la penalización por error en la clasificación
        # Si C tiene valores grandes entonces, se penaliza de forma más estricta los errores
        # si C es pequeño aumenta el sesgo y disminuye la varianza del modelo.
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }
    
    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print("="*64) 
        print(name)
        print("MSE: ", mean_squared_error(y_test, predictions))
        