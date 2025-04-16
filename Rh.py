# Importações
    # Bibliotécas de manipulação de dados

import pandas as pd
import numpy as np

    # Bibliotécas de visualização

import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno

    # Bibliotécas de estatística 

import scipy
from scipy.stats import normaltest
from scipy.stats import chi2_contingency

    # Bibliotécas de engenharia de atributos

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce

    # Bibliotécas para Ignore Warning

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

    # Carregando os dados

df = pd.read_csv("dataset/aug_train.csv")


print(df.shape)
print(df.columns)
print(df.head())
print(df.info())


# Análise exploratória dos dados

    # Dados não numéricos 

print(df.describe(include = object))

    # Dados numéricos

print(df.describe().drop(columns = ['enrollee_id', 'target']))

    # Variáveis categóricas

print(list(df.columns.values)[3:12])

# Plotagem

plt.figure(figsize= (18,30))
column_list = list(df.columns.values)[3:12]

A = 0

for i in column_list:
    A += 1
    plt.subplot(5, 2, A)
    ax = sns.countplot(data = df.fillna('NaN'), x = i)
    plt.title(i, fontsize = 15)
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.4, p.get_height()), ha = 'center', color = 'black', size = 12)
    if A >= 7:
        plt.xticks(rotation = 45)

plt.tight_layout(h_pad = 2)
plt.show()

# Distribuição das variáveis numéricas

print(df.describe().drop(columns = ['enrollee_id', 'target']))
plt.figure(figsize = (17,12))
plt.subplot(221)
sns.color_palette("hls", 8)
sns.histplot(df['city_development_index'], kde = True, color = "green")
plt.title('Histograma do CDI', fontsize = 20)

plt.subplot(222)
sns.histplot(df['training_hours'], kde = True, color = "magenta")
plt.title('Histograma das Horas de Treinamento', fontsize = 20)

plt.subplot(223)
sns.boxplot(df['city_development_index'], color = "green")
plt.subplot(224)
sns.boxplot(df['training_hours'], color = "magenta")
plt.show()


# Teste de Normalidade da Distribuição

numerical_feature = ['city_development_index', 'training_hours']

for i in numerical_feature:
    stats, pval = normaltest(df[i])
    if pval > 0.05:
        print(i, ': Distribuição Normal')
    else:
        print(i, ': Distribuição Não Normal')

print(df.head())
print(df.columns)

df_numerical = df.copy()
print(df_numerical["experience"].value_counts)

df_numerical["experience"] = np.where(df_numerical["experience"] == "<1", 1, df_numerical["experience"])
df_numerical["experience"] = np.where(df_numerical["experience"] == ">20", 21, df_numerical["experience"])
df_numerical["experience"] = df_numerical["experience"].astype(float)

print(df_numerical["experience"].value_counts())

print(df_numerical["last_new_job"].value_counts())

# Convertendo variável para numérica 

df_numerical["last_new_job"] = np.where(df_numerical["last_new_job"] == "never", 0, df_numerical["last_new_job"])
df_numerical["last_new_job"] = np.where(df_numerical["last_new_job"] == ">4", 5, df_numerical["last_new_job"])
df_numerical["last_new_job"] = df_numerical["last_new_job"].astype(float)

print(df_numerical["last_new_job"].value_counts())
print(df_numerical.head())
print(df_numerical.info())

df_corr = df_numerical.drop("enrollee_id", axis = 1).select_dtypes(include=["number"])
correlation_matrix = df_corr.corr(method="spearman")
print(correlation_matrix)

# Heatmap

plt.figure(figsize = (7,7))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
plt.title("Mapa de Correlação das Variáveis Numéricas\n", fontsize = 15)
plt.show()

for i in df.drop(columns = [ 'target',
                            'enrollee_id',
                            'city',
                            'city_development_index',
                            'training_hours',
                            'experience',
                            'last_new_job',
                            'company_size']).columns:
    df_woe_iv = (pd.crosstab(df[i], df['target'], normalize = 'columns')
                 .assign(woe = lambda dfx: np.log(dfx[1] / dfx[0]))
                 .assign(iv = lambda dfx: np.sum(dfx['woe'] * (dfx[1]-dfx[0]))))
    print(df_woe_iv,'\n-------------------------------------------------------------------')

# Variáveis categóricas

columns_cat = df.drop(columns = [ 'target',
                            'enrollee_id',
                            'city',
                            'city_development_index',
                            'training_hours',
                            'experience',
                            'last_new_job',
                            'company_size']).columns
iv = []

for i in columns_cat:
    df_woe_iv = (pd.crosstab(df[i], df['target'], normalize = 'columns')
                 .assign(woe = lambda dfx: np.log(dfx[1] / dfx[0]))
                 .assign(iv = lambda dfx: np.sum(dfx['woe']*(dfx[1]-dfx[0]))))
    iv.append(df_woe_iv['iv'][0])

df_iv = pd.DataFrame({'Features': columns_cat, 'iv':iv}).set_index('Features').sort_values(by = 'iv')

df_iv.plot(kind = 'barh', title = 'Information Value das Variáveis Categóricas', colormap = "Accent")
for index, value in enumerate(list(round(df_iv["iv"], 3))):
    plt.text((value), index, str(value))
plt.legend(loc = "lower right")
plt.show()

# Identificando Valores Ausentes

null_df = df.isna().sum().reset_index()
null_df.columns = ['Atributo', 'Total Ausentes']

ax = plt.figure(figsize = (15,5))
ax = sns.barplot(x='Atributo', y='Total Ausentes', data = null_df, palette = 'husl')

plt.xlabel('Atributos', fontsize = 12)
plt.ylabel('Contagem de Valores Ausentes', fontsize = 12)
plt.xticks(rotation = 45)
plt.title("Valores Ausentes", fontsize = 15)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha = 'center', va = 'bottom', fontsize = 11, color = 'black')
plt.show()

df_nan = pd.DataFrame(df.isna().sum())

if df.isna().any(axis = None):
    missingno.matrix(df[df_nan[df_nan[0]>0].index])
plt.show()

# Checando valores duplicados

df['enrollee_id'].duplicated().sum()

# Identificando Dados Desbalanceados

plt.figure(figsize = (17,(100)/ 20))

plt.subplot(121)

plt.pie(round(df['target'].value_counts() / len(df) * 100, 2),
        labels = list(df['target'].value_counts().index),
        autopct = "%.2f%%",
        explode = (0,0.1))

plt.axis("equal")
plt.title("Desequilíbrio da Variável Alvo", size = 15)

plt.subplot(122)
ax = sns.countplot(data = df, x = 'target')
plt.title("Distribuição da Variável Alvo", fontsize = 15)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}',
                (p.get_x()+0.4,
                 p.get_height()),
                 ha = 'center',
                 va = 'top',
                 color = 'white',
                 size = 12)
plt.show()

# Tratando Valores Ausentes

print(df.columns)
colunas_manter =['city_development_index',
                 'experience',
                 'enrolled_university',
                 'relevent_experience',
                 'education_level',
                 'company_type',
                 'major_discipline',
                 'target']
new_df = df[colunas_manter]

print(new_df.head())
print(df.head())

# Valores ausentes por coluna

null_df = new_df.isna().sum().reset_index()

ax = plt.figure(figsize = (15,6))

ax = sns.barplot(x = null_df['index'], y = null_df[0], palette = 'husl')
plt.xlabel('Atributos', fontsize = 12)
plt.ylabel('Contagem de Valores Ausentes', fontsize = 12)
plt.xticks(rotation = 45)
plt.title("Plot de Valores Ausentes", fontsize = 15)

for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, (p.get_height())), ha = 'center', color = 'black', size = 11)
plt.show()

sns.countplot(data = new_df.fillna('NaN'), x = 'major_discipline', alpha = 0.7, edgecolor = 'black')
plt.xticks(rotation = 45)
bound = ax.get_xbound()
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size = 10)
plt.title("Valores Ausentes da Variável major_discipline Antes do Processamento\n", fontsize = 15)
plt.show()

#Relação entre major_discipline e education_level

print('\nTotal de Valores Ausentes na Variável major_discipline: ', new_df['major_discipline'].isna().sum())
print('\nProporção de Valores Ausentes na Variável education_level: ') 
new_df[new_df['major_discipline'].isna()]['education_level'].value_counts(dropna = False)

# Preparando o índice

nan_index = (new_df[(new_df['major_discipline'].isna()) & ((new_df['education_level'] == 'High School') | (new_df['education_level'].isna()) | (new_df['education_level'] == 'Primary School'))]).index
print(len(nan_index))

# Preenchendo Valores Ausentes

new_df['major_discipline'][nan_index] = 'Non Degree'
print('Total de Valores Ausentes na Variável major_discipline: ', new_df['major_discipline'].isna().sum())
new_df['major_discipline'].value_counts(dropna = False)

# Valores Ausentes da Variável Após do Processamento

sns.countplot(data = new_df.fillna('NaN'), x = 'major_discipline', alpha = 0.7, edgecolor = 'black')
sns.despine()
plt.xticks(rotation=45)
bound=ax.get_xbound()
ax=plt.gca()
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size = 10)
plt.title(" Valores Ausentes da Variável major_discipline Após o Processamento\n", fontsize = 15)
plt.show()

print(new_df.head())

# Variável enrolled_university

sns.countplot(data = new_df.fillna('NaN'), x = 'enrolled_university', alpha = 0.7, edgecolor = 'black')
sns.despine()
plt.xticks()
bound=ax.get_xbound()
ax=plt.gca()
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size = 10)
plt.title("Valores Ausentes da Variável enrolled_university Antes do Processamento\n", fontsize = 15)
plt.show()

print('\nTotal de Valores Ausentes na Variável enrolled_university:', new_df['enrolled_university'].isna().sum())
print('\nProporção de Valores Ausentes na Variável education_level:')
new_df[new_df['enrolled_university'].isna()]['education_level'].value_counts(dropna = False)

# Preparando o índice
nan_index = (new_df[(new_df['enrolled_university'].isna()) & (new_df['education_level']=='Primary School')]).index

len(nan_index)

# Preenchendo Valores Ausentes
new_df['enrolled_university'][nan_index] = 'Primary Grad'

print('Total de Valores Ausentes:', new_df['enrolled_university'].isna().sum())
new_df[new_df['enrolled_university'].isna()]['education_level'].value_counts(dropna = False)

# Preparando o índice
nan_index = new_df[(new_df['enrolled_university'].isna())].index

# O restante coloco como 'Other'
new_df['enrolled_university'][nan_index] = 'Other'

sns.countplot(data = new_df.fillna('NaN'), x = 'enrolled_university', alpha = 0.7, edgecolor = 'black')
sns.despine()
plt.xticks()
bound=ax.get_xbound()
ax=plt.gca()
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size = 10)
plt.title("Valores Ausentes da Variável enrolled_university Após o Processamento\n", fontsize = 15)
plt.show()

print(new_df.head())

# Variável company_type

plt.figure(figsize = (20, 20))
column_list = ['company_type']
A = 0
for i in column_list:
    A+=1
    plt.subplot(4,2,A)
    ax = sns.countplot(data = new_df.fillna('NaN'), x = i, alpha = 0.7, edgecolor = 'black')
    sns.despine() 
    plt.title(i, fontsize = 15)
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size = 12)
    if A >=0:
        plt.xticks(rotation = 45)
plt.show()

new_df['company_type'].value_counts(dropna = False)

# Preparando o Índice
nan_index = new_df[(new_df['company_type'].isna())].index

# Preenchendo valores NaN com 'Other'
new_df['company_type'][nan_index] = 'Other'

plt.figure(figsize = (20, 20))
column_list = ['company_type']
A = 0
for i in column_list:
    A+=1
    plt.subplot(4,2,A)
    ax = sns.countplot(data = new_df.fillna('NaN'), x = i, alpha = 0.7, edgecolor = 'black')
    sns.despine() 
    plt.title(i, fontsize = 15)
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size = 12)
    if A >=0:
        plt.xticks(rotation = 45)
plt.show()

print(new_df.head())