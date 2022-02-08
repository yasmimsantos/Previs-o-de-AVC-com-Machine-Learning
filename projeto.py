'''
 ____________________________________________________________________________________________
|                                                                                            |
|                Projeto em Python da Disciplina Ambientes de Computação                     |  
|                    (CÓDIGO: DIP DCC909 01 AMBIENTES DE COMPUTAÇÃO)                         |
|                                                                                            |
|                 Título do Projeto: Previsão de AVC com Machine Learning                    |                                     
|                                                                                            |
|                            Yasmim Senden dos Santos 08/02/2022                             |
|                                                                                            |            
|     Descrição do Projeto: Aplicar o algoritmo de Machine Learning de Árvores de Decisão    |
|     para prever a chance de ter AVC ou não, com base em vários fatores.                    |
|____________________________________________________________________________________________|


'''
# O "sklearn" é uma biblioteca do Python utilizada para Machine Learning.
# O algoritmo "ExtraTrees" cria árvores de decisão.
# A função “ExtraTreesClassifier()” será utilizada pois o problema dos dados envolve CLASSIFICAÇÃO.
from sklearn.ensemble import ExtraTreesClassifier

# A função "train_test_split" é utilizada para distribuição de dados em dois grupos de forma aleatória.
from sklearn.model_selection import train_test_split

# Carrega um arquivo csv presente no mesmo diretório no qual o programa está.
import pandas as pd
arquivoAVC = pd.read_csv('testestroke.csv', delimiter=";")

# Imprime as primeiras 5 linhas
print(arquivoAVC.head())

# A variável "stroke" contém os dados que serão previstos, que estão na coluna stroke do arquivo csv.
stroke = arquivoAVC['stroke']

# A variável "restante" contém o restante das colunas do arquivo csv, que são as variáveis preditoras.
restante = arquivoAVC.drop('stroke', axis=1)

# Quando as variáveis stroke e restante são indicadas na função "train_test_split", significa que elas serão separadas em dois grupos aleatórios.
# O parâmetro “test_size = 0.3” determina que os dados de teste receberão 30% dos dados, e os dados de treino 70%.
restante_treino, restante_teste, stroke_treino, stroke_teste = train_test_split(
    restante, stroke, test_size=0.3)

print(arquivoAVC.shape, restante_treino.shape,
      restante_teste.shape, stroke_treino.shape, stroke_teste.shape)


# A função “fit” passa para o algoritmo as variáveis preditoras (restante) e a variável alvo (stroke), com o objetivo de que o algoritmo possa entender a relação entre estes dados e APRENDER a chegar a um modelo ideal.

modelo = ExtraTreesClassifier()
modelo.fit(restante_treino, stroke_treino)

# A função “score” passa para o algoritmo os dados de teste, para que seja possível avaliar o seu desempenho.
resultado = modelo.score(restante_teste, stroke_teste)
print("Previsão de acertos (%):", resultado)

# Exemplo: Selecionando aleatoriamente 10 amostras.
print("Amostras da coluna stroke: ", stroke_teste[100:110])

# É possível comparar com as amostras selecionadas acima e verificar se o algoritmo acertou em sua previsão.
previsoesDoAlgoritmo = modelo.predict(restante_teste[100:110])
print("Previsões do Algoritmo: ", previsoesDoAlgoritmo)
