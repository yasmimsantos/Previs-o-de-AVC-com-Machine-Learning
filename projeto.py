'''
____________________________________________________________________________________________
|                                                                                           |
|                                                                                           |
|                    Project Title: Stroke Prediction With Machine Learning                 |
|                                                                                           |
|                         Yasmim Senden dos Santos 08/02/2022                               |
|                                                                                           |
|          Project Description: Apply the Decision Trees Machine Learning Algorithm         |
|          to predict the chance of having a stroke or not, based on several factors.       |
|___________________________________________________________________________________________|


'''
# "sklearn" is a Python library used for Machine Learning.
# The "ExtraTrees" algorithm creates decision trees.
# The “ExtraTreesClassifier()” function will be used because the data problem involves CLASSIFICATION.
from sklearn.ensemble import ExtraTreesClassifier

# The "train_test_split" function is used to randomly distribute data into two groups.
from sklearn.model_selection import train_test_split

# Load a csv file present in the same directory where the program is.
import pandas as pd
arquivoAVC = pd.read_csv('testestroke.csv', delimiter=";")

# Print the first 5 lines
print(arquivoAVC.head())

# The "stroke" variable contains the data that will be predicted, which is in the stroke column of the csv file.
stroke = arquivoAVC['stroke']

# The "remaining" variable contains the rest of the columns of the csv file, which are the predictor variables.
restante = arquivoAVC.drop('stroke', axis=1)

# When stroke and remainder variables are indicated in the "train_test_split" function, it means that they will be separated into two random groups.
# The parameter “test_size = 0.3” determines that the test data will receive 30% of the data, and the training data 70%.
restante_treino, restante_teste, stroke_treino, stroke_teste = train_test_split(
    restante, stroke, test_size=0.3)

print(arquivoAVC.shape, restante_treino.shape,
      restante_teste.shape, stroke_treino.shape, stroke_teste.shape)


# The “fit” function passes to the algorithm the predictor variables (remaining) and the target variable (stroke), with the objective that the algorithm can understand the relationship between these data and LEARN to arrive at an ideal model.

modelo = ExtraTreesClassifier()
modelo.fit(restante_treino, stroke_treino)

# The “score” function passes the test data to the algorithm, so that its performance can be evaluated.
resultado = modelo.score(restante_teste, stroke_teste)
print("Previsão de acertos (%):", resultado)

# Example: Randomly selecting 10 samples.
print("Amostras da coluna stroke: ", stroke_teste[100:110])

# It is possible to compare with the samples selected above and verify that the algorithm was right in its prediction.
previsoesDoAlgoritmo = modelo.predict(restante_teste[100:110])
print("Previsões do Algoritmo: ", previsoesDoAlgoritmo)
