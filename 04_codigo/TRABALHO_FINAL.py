# PROJETO FINAL - MACHINE LEARNING
# Classificação: Ansiedade alta ou não

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# 1 - CARREGAR DADOS
df = pd.read_csv(r"C:\Users\Lessiee\Desktop\PROJETO FINAL\ansiedade_e_depressao.csv")

print(df.head())


# 2 - CRIAR VARIÁVEL ALVO
df["High_Anxiety"] = (df["Anxiety_Score"] >= 7).astype(int)

print(df["High_Anxiety"].value_counts())


# gráfico simples
df["High_Anxiety"].value_counts().plot(kind="bar")
plt.title("Ansiedade Alta vs Baixa")
plt.show()


# 3 - REMOVER COLUNA ORIGINAL
df = df.drop("Anxiety_Score", axis=1)


# 4 - TRANSFORMAR TEXTO EM NÚMEROS
df = pd.get_dummies(df)


# 5 - SEPARAR DADOS
X = df.drop("High_Anxiety", axis=1)
y = df["High_Anxiety"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# 6 - CRIAR E TREINAR MODELO
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)


# 7 - FAZER PREVISÕES
pred = modelo.predict(X_test)


# 8 - RESULTADOS
print("\nACURÁCIA:", accuracy_score(y_test, pred))
print("\nMATRIZ DE CONFUSÃO:\n", confusion_matrix(y_test, pred))


# 9 - EXPERIMENTOS SIMPLES

# Experimento 1: mais árvores
modelo1 = RandomForestClassifier(n_estimators=200)
modelo1.fit(X_train, y_train)
pred1 = modelo1.predict(X_test)

print("\nExperimento 1:", accuracy_score(y_test, pred1))


# Experimento 2: árvore mais limitada
modelo2 = RandomForestClassifier(max_depth=5)
modelo2.fit(X_train, y_train)
pred2 = modelo2.predict(X_test)

print("Experimento 2:", accuracy_score(y_test, pred2))


# Experimento 3: divisão diferente dos dados
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo3 = RandomForestClassifier()
modelo3.fit(X_train2, y_train2)
pred3 = modelo3.predict(X_test2)

print("Experimento 3:", accuracy_score(y_test2, pred3))