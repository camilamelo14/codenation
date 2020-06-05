#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[299]:


import pandas as pd
import numpy as np


# In[300]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[301]:


black_friday.head(5)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[302]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    pass 


# In[421]:


print(q1())
type(q1())


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[304]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')].shape[0]
    pass


# In[420]:


print(q2())
type(q2())


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[306]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.User_ID.nunique()
    pass


# In[423]:


print(q3())
type(q3())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[308]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.value_counts().size
    pass


# In[424]:


print(q4())
type(q4())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[330]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return np.float(((black_friday.isna().sum() / black_friday.shape[0]) * 100).max() / 100)
    pass


# In[311]:


#Avaliação
exploracao = pd.DataFrame({'nomes' : black_friday.columns, 'tipos' : black_friday.dtypes, 'NA #': black_friday.isna().sum(), 'NA %' : (black_friday.isna().sum() / black_friday.shape[0]) * 100})
exploracao


# In[425]:


print(q5())
type(q5())


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[343]:


def q6():
    # Retorne aqui o resultado da questão 6.
    exploracao = pd.DataFrame({'nomes' : black_friday.columns, 'tipos' : black_friday.dtypes, 'NA #': black_friday.isna().sum(), 'NA %' : (black_friday.isna().sum() / black_friday.shape[0]) * 100})
    return np.int(exploracao['NA #'].max())
    pass


# In[426]:


print(q6())
type(q6())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[262]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return np.float(black_friday.Product_Category_3.mode())
    pass


# In[427]:


print(q7())
type(q7())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[333]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return black_friday.Purchase.mean() / 10000
    pass


# In[402]:


#black_friday.Purchase.fillna(black_friday.Purchase.mean(), inplace=True)
#black_friday.Purchase.mean()
black_friday.Purchase.count()


# In[428]:


print(q8())
type(q8())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[408]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return np.int(((black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()).between(-1,1).sum())
    pass


# In[429]:


print(q9())
type(q9())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[232]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return np.bool(black_friday['Product_Category_2'].isna().count() == black_friday['Product_Category_3'].isna().count())
    pass


# In[430]:


print(q10())
type(q10())

