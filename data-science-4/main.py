#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv",decimal=',')


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# In[5]:


df = countries
df.info()


# In[6]:


df['Country'] = df.Country.str.strip()
df['Region'] = df.Region.str.strip()


# In[7]:


df.describe()


# ## Inicia sua análise a partir daqui

# In[8]:


# Sua análise começa aqui
df['Region'] = df.Region.sort_values()
unique_regions_sorted = df.Region.unique()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    # Retorne aqui o resultado da questão 1.
    df['Region'] = df.Region.sort_values()
    unique_regions_sorted = df.Region.unique().tolist()
    return sorted(unique_regions_sorted)
    pass


# In[10]:


print(q1())
print(type(q1()))


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[11]:


def q2():
    # Retorne aqui o resultado da questão 2.
    #n_bins eh o parametro com os intervalos definidos, se faz o fit e trans para aplicá-lo 
    disct = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    disct.fit(countries[['Pop_density']])
    score_bins = disct.transform(df[['Pop_density']])
    
    #Aplica a regra para encontrar a quantidade de paises acima de 90º percentil (i>8)
    num_countries = [sum(score_bins[:, 0] == i) for i in range(len(disct.bin_edges_[0])-1) if i>8]
    return np.int(num_countries[0])
    pass


# In[12]:


print(q2())
print(type(q2()))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[13]:


def q3():
    # Retorne aqui o resultado da questão 3.
    aux = df
    result = pd.get_dummies(aux[['Region', 'Climate']].fillna(''))
    return int(result.shape[1])
    pass


# In[14]:


print(q3())
print(type(q3()))


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[15]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[16]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return 0.000
    pass


# In[17]:


print(q4())
print(type(q4()))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[18]:


def q5():
    # Retorne aqui o resultado da questão 4.
    outlier_eval = df.Net_migration.copy()

    q1 = outlier_eval.quantile(0.25)
    q3 = outlier_eval.quantile(0.75)
    iqr = q3 - q1

    no_outlier = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
    return (np.int(outlier_eval[outlier_eval < no_outlier[0]].count()), np.int(outlier_eval[outlier_eval > no_outlier[1]].count()), False)
    pass


# In[19]:


print(q5()[0])
print(type(q5()[0]))


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[20]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[21]:


def q6():
    # Retorne aqui o resultado da questão 4.
    c_vector = CountVectorizer()
    result = c_vector.fit_transform(newsgroup.data)
    return np.int(result[:, c_vector.vocabulary_['phone']].sum())
    pass


# In[22]:


print(q6())
print(type(q3()))


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[23]:


def q7():
    # Retorne aqui o resultado da questão 4.
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(newsgroup.data)
    
    result = tfidf_vect.transform(newsgroup.data)
    return round(result[:, tfidf_vect.vocabulary_['phone']].sum(),3)
    pass


# In[24]:


print(q7())
print(type(q7()))

