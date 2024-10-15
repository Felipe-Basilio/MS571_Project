import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import openpyxl

players = pd.read_csv("players.csv")

picks = pd.read_csv("picks.csv")

resultados = pd.read_csv("results.csv")

#######

#Avaliação dados - Players

print(players.head())

print(players.info())

print(players.describe())

#Avaliação dados - Picks

print(picks.head())

print(picks.info())

print(picks.describe())

#Avaliação dados - Results

print(resultados.head())

print(resultados.info())

print(resultados.describe())
