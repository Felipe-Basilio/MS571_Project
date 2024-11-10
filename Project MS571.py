import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Cartão de Crédito/creditcard.csv")

data.head()

data.describe()

#Ver se tem valores faltantes

data.isnull().sum().max()

#Nome das colunas
data.columns

#Contagem de Fraudes / Não Fraudes

print('Fraudes', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% no dataset')
print('Não Fraudes', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% no dataset')


fraud_counts = data['Class'].value_counts()

#Percebemos que os dados estão desbalanceados e verificamos também nos gráficos

plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=["Não Fraude", "Fraude"], autopct='%1.1f%%', colors=['blue', 'red'])
plt.title("Distribuição das Classes (Fraude vs Não Fraude)")
plt.show()

#Distribuição das transações ao longo do tempo e ao longo da quantidade

fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribuição das Transações Quantidade', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribuição das Transações Tempo', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
plt.show()

#precisamos criar uma subamostra do dataframe para obter uma quantidade igual de casos de Fraude e Não-Fraude, ajudando nossos algoritmos a entender melhor os padrões que determinam se uma transação é fraudulenta ou não.
#O que é uma Subamostra?
#Neste contexto, nossa subamostra será um dataframe com uma proporção de 50/50 de transações fraudulentas e não fraudulentas. Isso significa que nossa subamostra terá a mesma quantidade de transações de fraude e de não fraude.
#Por que criamos uma Subamostra?
#No início deste notebook, vimos que o dataframe original estava fortemente desbalanceado! Usar o dataframe original causaria os seguintes problemas:
#Overfitting (Ajuste Excessivo): Nossos modelos de classificação assumiriam que, na maioria dos casos, não há fraudes! O que queremos para o nosso modelo é que ele seja preciso ao identificar uma fraude.
#Correlações Incorretas: Embora não saibamos o que representam as variáveis "V", seria útil entender como cada uma dessas características influencia o resultado (Fraude ou Não Fraude). Com um dataframe desbalanceado, não conseguimos observar as correlações reais entre as classes e as variáveis.

from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

print(data.head())

#Antes de prosseguir com o undersampling, vamos realizar a separação dos conjuntos de teste e treinamento, visando o objetivo de testar o modelo com dataframes subamostrados ou superamostrados (para que nossos modelos possam detectar os padrões) e testá-lo no conjunto de teste original.

from sklearn.model_selection import StratifiedKFold
import numpy as np

# Exibir a distribuição das classes no dataset
print('Transações legítimas:', round(data['Class'].value_counts()[0] / len(data) * 100, 2), '% do dataset')
print('Fraudes:', round(data['Class'].value_counts()[1] / len(data) * 100, 2), '% do dataset')

# Separar as variáveis de entrada e de saída
X = data.drop('Class', axis=1)
y = data['Class']

# Configurar a divisão estratificada
sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Iterar sobre as divisões para criar conjuntos de treino e teste
for train_index, test_index in sss.split(X, y):
    print("Índices de treino:", train_index, "Índices de teste:", test_index)
    X_train_original, X_test_original = X.iloc[train_index], X.iloc[test_index]
    y_train_original, y_test_original = y.iloc[train_index], y.iloc[test_index]

# Converter os DataFrames para arrays numpy
X_train_original = X_train_original.values
X_test_original = X_test_original.values
y_train_original = y_train_original.values
y_test_original = y_test_original.values

# Verificar se a distribuição das classes nos conjuntos de treino e teste são similares
train_label_distribution = np.unique(y_train_original, return_counts=True)[1] / len(y_train_original)
test_label_distribution = np.unique(y_test_original, return_counts=True)[1] / len(y_test_original)

print('-' * 100)
print('Distribuição de Rótulos: \n')
print('Treino:', train_label_distribution)
print('Teste:', test_label_distribution)

#Nesta etapa do projeto, aplicaremos a técnica de Random Under Sampling (subamostragem aleatória), que visa reduzir a quantidade de dados na classe majoritária para equilibrar o dataset e diminuir o risco de overfitting nos modelos.
#Primeiro, vamos analisar o grau de desbalanceamento da classe, usando value_counts() na coluna de classe para ver a quantidade de instâncias de cada rótulo.
#Após identificar o número de transações fraudulentas (Fraude = "1"), ajustaremos o número de transações legítimas para igualá-lo (com uma proporção de 50/50). Assim, ficaremos com 492 casos de fraudes e 492 casos de transações legítimas.
#Com isso, teremos uma subamostra do dataset com proporção balanceada entre as classes. O próximo passo será embaralhar esses dados para verificar se o modelo mantém uma precisão consistente a cada execução do script.
#Ao fazer isso pode ocorrer perda de informações 

data = data.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = data.loc[data['Class'] == 1]
non_fraud_df = data.loc[data['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
novo_dataframe = normal_distributed_df.sample(frac=1, random_state=42)

print(novo_dataframe.head())

#Após o balanceamento dos dados 

print('Fraudes', round(novo_dataframe['Class'].value_counts()[1]/len(novo_dataframe) * 100,2), '% no dataset')
print('Não Fraudes', round(novo_dataframe['Class'].value_counts()[0]/len(novo_dataframe) * 100,2), '% no dataset')

fraud_counts = novo_dataframe['Class'].value_counts()

#Percebemos que os dados estão desbalanceados e verificamos também nos gráficos

plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=["Não Fraude", "Fraude"], autopct='%1.1f%%', colors=['blue', 'red'])
plt.title("Distribuição das Classes (Fraude vs Não Fraude)")
plt.show()

#Agora fazendo um mapa de matrizes de correlação

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Mapa de Correlação do DataFrame completo
corr = data.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

# Mapa de Correlação do DataFrame balanceado
sub_sample_corr = novo_dataframe.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()

#Focos de atenção nos mapas de correlação 
#Forte correlação positiva em V2, V4, V11 e V19
#Correlação negativa interessante em V10, V12, V14 e V17

f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Correlações positivas (Quanto maior o valor da característica, maior a probabilidade de ser uma transação fraudulenta)
sns.boxplot(x="Class", y="V2", data=novo_dataframe, palette="muted", ax=axes[0])
axes[0].set_title('V2 vs Class Correlação Positiva')

sns.boxplot(x="Class", y="V4", data=novo_dataframe, palette="muted", ax=axes[1])
axes[1].set_title('V4 vs Class Correlação Positiva')


sns.boxplot(x="Class", y="V11", data=novo_dataframe, palette="muted", ax=axes[2])
axes[2].set_title('V11 vs Class Correlação Positiva')


sns.boxplot(x="Class", y="V19", data=novo_dataframe, palette="muted", ax=axes[3])
axes[3].set_title('V19 vs Class Correlação Positiva')

plt.show()

f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Correlações negativas com nossa Classe (Quanto menor o valor da característica, mais provável que seja uma transação fraudulenta)
sns.boxplot(x="Class", y="V17", data=novo_dataframe, palette="muted", ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=novo_dataframe, palette="muted", ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V12", data=novo_dataframe, palette="muted", ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V10", data=novo_dataframe, palette="muted", ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()

from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = novo_dataframe['V14'].loc[novo_dataframe['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = novo_dataframe['V12'].loc[novo_dataframe['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = novo_dataframe['V10'].loc[novo_dataframe['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()

#V14 se assemelha a uma distribuição gaussiana se comparada ao resto 

#Aplicando a técnica de t-SNE e PCA para melhor visualização da estrutura dos dados

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Criar uma amostra balanceada (caso ainda não tenha feito)
# Exemplo: novo_dataframe deve ter uma quantidade balanceada de fraudes e não fraudes
X = novo_dataframe.drop("Class", axis=1)
y = novo_dataframe["Class"]

# Redução de Dimensionalidade com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualização da redução de dimensionalidade com PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Não Fraude", alpha=0.5, c="blue")
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Fraude", alpha=0.5, c="red")
plt.title("Redução de Dimensionalidade com PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.show()

# Redução de Dimensionalidade com t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(X)

# Visualização da redução de dimensionalidade com t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label="Não Fraude", alpha=0.5, c="blue")
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label="Fraude", alpha=0.5, c="red")
plt.title("Redução de Dimensionalidade com t-SNE")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.legend()
plt.show()

# Subamostragem antes da validação cruzada (propenso a overfitting)
X = novo_dataframe.drop('Class', axis=1)
y = novo_dataframe['Class']

# Nossos dados já estão escalonados; agora devemos dividir os conjuntos de treino e teste
from sklearn.model_selection import train_test_split

# Este é utilizado especificamente para subamostragem.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformar os valores em arrays para alimentar os algoritmos de classificação.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Vamos implementar classificadores simples

classifiers = {
    "LogisticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
}

from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

#O classificador de regressão logística é o mais acurado


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Logistic Regression com GridSearchCV
log_reg_params = {
    "penalty": ['l1', 'l2'], 
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "solver": ['liblinear']
}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, cv=5, scoring='accuracy')
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2), '%')


# K-Nearest Neighbors com GridSearchCV
knears_params = {
    "n_neighbors": list(range(2, 5, 1)), 
    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params, cv=5, scoring='accuracy')
grid_knears.fit(X_train, y_train)
knears_neighbors = grid_knears.best_estimator_

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('K-Nearest Neighbors Cross Validation Score:', round(knears_score.mean() * 100, 2), '%')


# Support Vector Classifier com GridSearchCV
svc_params = {
    "C": [0.5, 0.7, 0.9, 1], 
    "kernel": ['rbf', 'poly', 'sigmoid', 'linear']
}
grid_svc = GridSearchCV(SVC(), svc_params, cv=5, scoring='accuracy')
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score:', round(svc_score.mean() * 100, 2), '%')


# Decision Tree Classifier com GridSearchCV
tree_params = {
    "criterion": ["gini", "entropy"], 
    "max_depth": list(range(2, 4, 1)), 
    "min_samples_leaf": list(range(5, 7, 1))
}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=5, scoring='accuracy')
grid_tree.fit(X_train, y_train)
tree_clf = grid_tree.best_estimator_

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('Decision Tree Classifier Cross Validation Score:', round(tree_score.mean() * 100, 2), '%')

# We will undersample during cross validating
undersample_X = data.drop('Class', axis=1)
undersample_y = data['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print("Train:", train_index, "Test:", test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values 

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

# Implementing NearMiss Technique 
# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)
X_nearmiss, y_nearmiss = NearMiss().fit_resample(undersample_X.values, undersample_y.values)
print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

# Cross Validating the right way

for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    
    undersample_accuracy.append(undersample_pipeline.score(X_train_original[test], y_train_original[test]))
    undersample_precision.append(precision_score(y_train_original[test], undersample_prediction))
    undersample_recall.append(recall_score(y_train_original[test], undersample_prediction))
    undersample_f1.append(f1_score(y_train_original[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(y_train_original[test], undersample_prediction))

# Importando bibliotecas adicionais para o aprendizado de curva
from sklearn.model_selection import ShuffleSplit, learning_curve

# Função para plotar a curva de aprendizado
def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
        
    # Logistic Regression Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")

    # K-Nearest Neighbors Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
    ax2.set_title("K-Nearest Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")

    # Support Vector Classifier Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
    ax3.set_title("Support Vector Classifier Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")

    # Decision Tree Classifier Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")

    return plt

# Chamando a função com os estimadores otimizados
cv = StratifiedKFold(n_splits=5)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, cv=cv)
plt.show()

from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                             method="decision_function")
tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

from sklearn.metrics import roc_auc_score

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)

def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()

#Regressão Logística

def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12,8))
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
logistic_roc_curve(log_fpr, log_tpr)
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)

undersample_y_score = log_reg.decision_function(X_test_original)

from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(y_test_original, undersample_y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      undersample_average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(y_test_original, undersample_y_score)

plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('UnderSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          undersample_average_precision), fontsize=16)

#SMOTE Technique (Over-Sampling):

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Parâmetros do modelo para RandomizedSearchCV
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Realizando a busca de hiperparâmetros uma vez e fora do loop
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4, cv=3, scoring='accuracy', n_jobs=-1)
rand_log_reg.fit(X_train_original, y_train_original)
best_log_reg = rand_log_reg.best_estimator_

print('Melhor modelo Logistic Regression com SMOTE:', best_log_reg)

# Listas para armazenar as métricas
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Validando usando StratifiedKFold
for train, test in sss.split(X_train_original, y_train_original):
    # Pipeline com SMOTE usando o melhor modelo obtido do RandomizedSearch
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), best_log_reg)
    best_est = rand_log_reg.best_estimator_
    pipeline.fit(X_train_original[train], y_train_original[train])
    prediction = pipeline.predict(X_train_original[test])
    
    # Calculando as métricas
    accuracy_lst.append(pipeline.score(X_train_original[test], y_train_original[test]))
    precision_lst.append(precision_score(y_train_original[test], prediction))
    recall_lst.append(recall_score(y_train_original[test], prediction))
    f1_lst.append(f1_score(y_train_original[test], prediction))
    auc_lst.append(roc_auc_score(y_train_original[test], prediction))
    
# Resultados médios das métricas
print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print("auc: {}".format(np.mean(auc_lst)))
print('---' * 45)

labels = ['No Fraud', 'Fraud']
smote_prediction = best_est.predict(X_test_original)
print(classification_report(y_test_original, smote_prediction, target_names=labels))

y_score = best_est.decision_function(X_test_original)
average_precision = average_precision_score(y_test_original, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(y_test_original, y_score)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          average_precision), fontsize=16)

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train) 

# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_resample(X_train_original, y_train_original)

# We Improve the score by 2% points approximately 
# Implement GridSearchCV and the other models.

# Logistic Regression
t0 = time.time()
log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm.fit(Xsm_train, ysm_train)
t1 = time.time()
print("Fitting oversample data took :{} sec".format(t1 - t0))

#Test Data with Logistic Regression:

from sklearn.metrics import confusion_matrix

# Logistic Regression fitted using SMOTE technique
y_pred_log_reg = log_reg_sm.predict(X_test)

# Other models fitted with UnderSampling
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)


log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)

fig, ax = plt.subplots(2, 2,figsize=(22,12))


sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)
ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)
ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)
ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)


plt.show()

from sklearn.metrics import classification_report


print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))

print('KNears Neighbors:')
print(classification_report(y_test, y_pred_knear))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_svc))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_tree))

# Final Score in the test set of logistic regression
from sklearn.metrics import accuracy_score

# Logistic Regression with Under-Sampling
y_pred = log_reg.predict(X_test)
undersample_score = accuracy_score(y_test, y_pred)



# Logistic Regression with SMOTE Technique (Better accuracy with SMOTE t)
y_pred_sm = best_est.predict(X_test_original)
oversample_score = accuracy_score(y_test_original, y_pred_sm)


d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'], 'Score': [undersample_score, oversample_score]}
final_df = pd.DataFrame(data=d)

# Move column
score = final_df['Score']
final_df.drop('Score', axis=1, inplace=True)
final_df.insert(1, 'Score', score)

# Note how high is accuracy score it can be misleading! 
final_df
