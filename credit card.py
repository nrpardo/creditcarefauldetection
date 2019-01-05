
# coding: utf-8

# # https://www.kaggle.com/c/GiveMeSomeCredit/data    

# Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years.
# 
# Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 
# 
# Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.
# 
# The goal of this competition is to build a model that borrowers can use to help make the best financial decisions.
# 

# In[6]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# In[7]:


creditos_df = pd.read_csv("data/datos_creditos.csv")


# In[ ]:


creditos_df.head()


# In[ ]:


creditos_df["impago_en_2_anos"].value_counts()
#el dataset se encuentra inbalanceado.


# In[ ]:


#nuestra variable objetivo será
variable objetivo= "impago_en_2_anos"


# In[ ]:


X = creditos_df.drop(variable_objetivo, axis=1)
y = creditos_df[variable_objetivo]

X_train_credito, X_test_credito, y_train_credito, y_test_credito = train_test_split(
    X, y, test_size=0.3, random_state=42)


# In[ ]:


y.value_counts(normalize=True)


# In[ ]:


modelo = LogisticRegression()

modelo.fit(X_train_credito, y_train_credito)

predicciones_creditos = modelo.predict(X_test_credito)
clases_reales_creditos = y_test_credito
predicciones_probabilidades_creditos = modelo.predict_proba(X_test_credito)


# In[ ]:


evaluar_modelo(clases_reales_creditos, predicciones_creditos, predicciones_probabilidades_creditos)


# In[ ]:


len(creditos_df[creditos_df[variable_objetivo]==1])


# In[ ]:


len([pred for pred in predicciones if pred==1])


# In[ ]:


grafica_precision_recall(clases_reales_creditos, predicciones_probabilidades_creditos)


# In[ ]:


grafica_curva_auc(clases_reales_creditos, predicciones_creditos, predicciones_probabilidades_creditos)

