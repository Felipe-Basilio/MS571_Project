import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import openpyxl
import matplotlib.pyplot as plt

dados = pd.read_csv("Cartão de Crédito/creditcard.csv")

def split_train_val_test(data, test_ratio, val_ratio, random_state=42):
    np.random.seed(random_state)  # Fixa a seed para reprodutibilidade
    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)
    val_set_size = int(len(data) * val_ratio)

    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    val_indices = shuffled_indices[test_set_size:test_set_size + val_set_size]

    return data.iloc[train_indices], data.iloc[val_indices], data.iloc[test_indices]

train_set, val_set, test_set = split_train_val_test(dados, val_ratio=0.2, test_ratio=0.2)

# Salvar os conjuntos de treino, validação e teste em arquivos CSV
train_set.to_csv('train_set.csv', index=False)
val_set.to_csv('val_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)

#Modelo de Regressão Linear

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit()
