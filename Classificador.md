O código estará dividido por seções para maior organização. 
Como primeiro passo, irei fazer a importação de todas as bibliotecas que irei usar e em seguida preparar e limpar os dados para manter aqueles que são interessantes para o nosso objetivo.

Estou fazendo uso da função stopwords que contém palavras básicas que serão removidas da base.

Já que o objetivo e separar uma review como sendo 'positiva' ou 'negativa' foi escolhida uma análise de sentimento para processar todo o histórico de reviews e quantificar a opinião dos clientes, o que nos
permitiria monitorar a satisfação dos clientes ao longo do tempo e os pontos positivos e negativos que precisam destaque e/ou melhoria.

```python
# =============================================================================
# SEÇÃO 1: IMPORTAÇÃO DAS BIBLIOTECAS, PREPARAÇÃO E LIMPEZA DOS DADOS
# =============================================================================
## Carregamento e preparação dos dados
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
stopwords = stopwords.words('portuguese')

data = pd.read_csv('sample_data/olist_order_reviews_dataset.csv')

#Aqui mantemos somente as colunas que são interessantes para nós
df_model = data[['review_score', 'review_comment_message']].copy()
#E aqui removemos as linhas que não contém comentários escritos nas suas avaliações
df_model.dropna(subset=['review_comment_message'], inplace=True)

#Criamos uma função onde vamos a converter a nota em sentiment

def to_sentiment(score): 
  score = int(score)
  if score <= 2:
    return 'negative'
  elif score >= 4:
    return 'positive'
  else:
    return 'neutral'


#Criaremos a coluna de sentimento com base a sua review score

df_model['sentiment'] = df_model['review_score'].apply(to_sentiment)

#Filtramos fora as avaliações 3 estrelas, que estamos considerando como neutras

df_final = df_model[df_model['sentiment'] != 'neutral'].copy()

print("Preparação dos dados concluída.")
print(f"Total de amostras para o modelo: {len(df_final)}")
print(df_final['sentiment'].value_counts())

# Define as variáveis X (features, o texto) e y (target, o sentimento)
X = df_final.review_comment_message
y = df_final.sentiment

- - - - - - - - - - - - - -  OUTPUT - - - - - - - - - - - - - - 
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
Preparação dos dados concluída.
Total de amostras para o modelo: 37420
sentiment
positive    26530
negative    10890
Name: count, dtype: int64


```

Como seguinte passo iremos separar nosso dados para treino e validação (80%) e os restantes (20%) serão usados para o teste final
e o pre processamento nos ajudará com a normalização e limpeza do texto

```python
# =============================================================================
# SEÇÃO 2: DIVISÃO EM DADOS DE TREINO/VALIDAÇÃO E TESTE
# =============================================================================
# Separa 20% dos dados para o teste final
# Os 80% restantes serão usados para treino e cross validation
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# usaremos uma função de pre-processamento para limpar o texto

preprocessor = lambda text: re.sub(r'[^a-zÀ-ú ]', '', str(text).lower())
```
Para nossa seção 3 chegamos na construção do pipeline, a pesar de ter o método que irei usar definido eu implementei um grid search para poder testar
varios modelos com diferentes parametros.

Para a construção do pipeline, ele será dividido em 3 etapas: 
- Primeiro iremos transformar o texto em vetores numéricos, onde cada número representa a contagem de uma palavra cujos parametros seriam o preprocessador, stopwords e min_df que irá dar uma frequência mínima de documento.
- Depois passará por TfidfTransformer o que ajudará o modelo a entender quais palavras são mais importantes e distintas em cada avaliação em vez de só focar na sua frequência
- Finalmente as pontuações serão recebidas pelo nosso algoritmo que aprenderá a ser classificar o texto;

Para o Gridsearch optei por testar diferentes valores para O SGDClassifier e também testar 2 modelos diferentes com o uso dele, regressão logística e SVM Linear.
Também optei por fazer testes com um classificador clássico Naive Bayes que costuma ser eficaz para texto.

Ao fazer todas as modificações o melhor modelo de todos será salvo na variavel best_model onde ele será testado com os 20% de dados restantes para obter uma avaliaçãio final da sua performance.

```python
# SEÇÃO 3: CONSTRUÇÃO DO PIPELINE E OTIMIZAÇÃO USANDO GRIDSEARCHCV
# =============================================================================
# Função de pré-processamento para limpar o texto (remover números, pontuação, etc.)
from sklearn.model_selection import train_test_split, GridSearchCV

# montando o pipeline
pipe = Pipeline([
    ('vec', CountVectorizer(stop_words=stopwords, min_df = 5, preprocessor=preprocessor)), #etapa de vectorização
    ('tfid', TfidfTransformer()),
    ('clf', SGDClassifier(random_state=42)) # garantindo que os resultados do modelo sejam sempre os mesmos com random state definido
])

# Grid de busca para o SGDClassifier
param_grid_sgd = {
    'vec__min_df': [3,5,7,10],
    'vec__ngram_range': [(1, 1)],
    'clf': [SGDClassifier(random_state=42)], # <-- O modelo em si!
    'clf__loss': ['log_loss', 'hinge'],      # <-- Testa Regressão Logística vs. SVM
    'clf__penalty': ['l2', 'l1'],
    'clf__alpha': [1e-2, 5e-3,1e-5]
}

from sklearn.naive_bayes import MultinomialNB

# Grid de busca para o MultinomialNB
param_grid_nb = {
    'vec__ngram_range': [(1, 1), (1, 2)],
    'clf': [MultinomialNB()],                  # usamos diferente modelo
    'clf__alpha': [0.1, 0.5, 1.0]              # parâmetro 'alpha' (smoothing) pertencente ao método Naive Bayes
}

#  criamos uma lista de dicionarios com ambos para inserir no nosso grid search
search_space = [
    param_grid_sgd,
    param_grid_nb
]

# O GridSearchCV com uma lista de grids
grid_search = GridSearchCV(pipe, search_space, cv=5, n_jobs=-1, verbose=1)

print("Iniciando a busca pelo melhor MODELO e seus parâmetros...")

# Usando os dados de treino/validação
grid_search.fit(X_train_val, y_train_val)

print("\nMelhor combinação de modelo e parâmetros encontrada:")
print(grid_search.best_params_)

# Avaliamos o melhor modelo conjunto de teste
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nRelatório de Classificação do MELHOR MODELO no conjunto de TESTE:")
print(classification_report(y_test, y_pred))

- - - - - - - - - - - - - -  OUTPUT - - - - - - - - - - - - - - 

Iniciando a busca pelo melhor MODELO e seus parâmetros...
Fitting 5 folds for each of 54 candidates, totalling 270 fits

Melhor combinação de modelo e parâmetros encontrada:
{'clf': SGDClassifier(random_state=42), 'clf__alpha': 1e-05, 'clf__loss': 'log_loss', 'clf__penalty': 'l2', 'vec__min_df': 7, 'vec__ngram_range': (1, 1)}

Relatório de Classificação do MELHOR MODELO no conjunto de TESTE:
              precision    recall  f1-score   support

    negative       0.87      0.85      0.86      2191
    positive       0.94      0.95      0.94      5293

    accuracy                           0.92      7484
   macro avg       0.91      0.90      0.90      7484
weighted avg       0.92      0.92      0.92      7484

```
Como podemos ver no output acima, obtivemos resultados eficazes, com um modelo com uma acurácia de 92%, ele está demonstrando uma dificuldade maior ao identificar reviews negativas, que era esperado devido a que a classe majoritária nos dados são reviews positivos, mas afinal representa um resultado robusto para nossos fins.


Como seguinte passo, temos a seção 4, agora que tenho o melhor modelo dentro os parâmetros informados, gostariamos de diagnosticar se existe overfitting nesses dados para podermos avançar com ele ou não,
já que é muito improtante sabermos se ele manterá esses resultados para outros dados fora do ambiente atual, para isto, será apresentado comparar as acurácias entre o treino e o teste para poder observar o gap
que existe entre elas.

```python
# =============================================================================
# SEÇÃO 4: AVALIAÇÃO FINAL DO MELHOR MODELO
# =============================================================================
print("\n--- AVALIAÇÃO FINAL DO MELHOR MODELO ---")

# Agora queremos checar se existe overfitting então iremos gerar previsões para os dados de treino e teste para checar
y_pred_train = best_model.predict(X_train_val)
y_pred_test = best_model.predict(X_test)

# Calcula e exibe as acurácias de treino e teste
accuracy_train = accuracy_score(y_train_val, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("\n" + "="*50)
print(f"Acurácia no conjunto de TREINO: {accuracy_train:.4f}")
print(f"Acurácia no conjunto de TESTE:  {accuracy_test:.4f}")
print("="*50)

# Exibe o relatório de classificação completo para o conjunto de teste
print("\nRelatório de Classificação no conjunto de TESTE:")
print(classification_report(y_test, y_pred_test))

- - - - - - - - - - - - - -  OUTPUT - - - - - - - - - - - - - - 

--- AVALIAÇÃO FINAL DO MELHOR MODELO ---

==================================================
Acurácia no conjunto de TREINO: 0.9357
Acurácia no conjunto de TESTE:  0.9200
==================================================

Relatório de Classificação no conjunto de TESTE:
              precision    recall  f1-score   support

    negative       0.87      0.85      0.86      2191
    positive       0.94      0.95      0.94      5293

    accuracy                           0.92      7484
   macro avg       0.91      0.90      0.90      7484
weighted avg       0.92      0.92      0.92      7484
```
Mediante os modelos estavam sendo adaptados, usei a estratégia de tentativa e erro para poder achar o melhor modelo, no entanto, alguns modelos mostravam uma acurácia maior do que os dados mostrados agora, porém,
na etapa de overfitting foi identificada um maior gap entre a acurácia de treino e teste, o que mostrou uma maior chance de overfitting acontecendo, pelo qual decidi reduzir um pouco a performance geral do modelo para poder obter um gap menor. Essa decisão foi tomada pelo seguinte motivo:

- O objetivo principal é poder identificar se uma review é positiva ou negativa, isto pode ajudar a tomar decisões estratégicas de pontos que devem ser abordados pela empresa e pontos que o consumidor se encontra
satisfeito, o que nos direciona a uma melhor toma de decisões para evitar recursos mal alocados, um overfitting menor nos permitirá uma maior confiança e robustez.


Na seção 5, iremos gerar a matriz de confusão para identificar onde exatamente o modelo está acertando e errando, com ajuda da uma função de Scikit confusion_matrix e uma função de seaborn para criar mapa de calor 
que torna a visualização muito mais fácil.

Também, iremos criar um dataframe onde mostraremos as 20 palavras mais importantes para uma review ser classificada como positiva ou negativa, mostrando a palavra e seu respectivo peso em seguida.
```python

# =============================================================================
# SEÇÃO 5: ANÁLISE PROFUNDA DO MELHOR MODELO
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Gerar matriz de confusão para checar onde o modelo está errando mais
print("\nGerando Matriz de Confusão para o conjunto de teste...")
cm = confusion_matrix(y_test, y_pred_test, labels=best_model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title('Matriz de Confusão do Melhor Modelo')
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.show()

# Também queremos ver o peso que as palavras têm para o modelo avaliar se é review positive ou negative
print("\nAnalisando as palavras mais importantes para o modelo...")
vectorizer = best_model.named_steps['vec']
classifier = best_model.named_steps['clf']
feature_names = np.array(vectorizer.get_feature_names_out())

# Acessa os coeficientes (pesos) que o modelo atribuiu a cada palavra

coeficientes = classifier.coef_[0]
df_importancia = pd.DataFrame({'palavra': feature_names, 'peso': coeficientes})

print("\n--- 20 palavras mais importantes para o sentimento POSITIVO ---")
display(df_importancia.sort_values(by='peso', ascending=False).head(20))

print("\n--- 20 palavras mais importantes para o sentimento NEGATIVO ---")
display(df_importancia.sort_values(by='peso', ascending=True).head(20))

- - - - - - - - - - - - - -  OUTPUT - - - - - - - - - - - - - -

```
/COLOCAR ANALISES SEÇÃO 5 AQUI/


Como seguinte passo agora resta identificar os motivos que tornam uma review positiva ou negativa, para isto, optei por uma análise de n-gramas
onde preferi manter ngram como sendo entre 3 e 4 já que irá identificar ideias completas que mostram claramente o motivo pelo qual é positivo ou negativo. 
A função irá encontrar todas as frases únicas e mostrar quantas vezes cada frase aparece em cada avaliação e mostraremos somente as top 15 frases mais comuns usadas como positivo ou negativo.

```python
# =============================================================================
# SEÇÃO 6: IDENTIFICAÇÃO DE MOTIVOS
# =============================================================================

# Separa os textos em duas listas: uma para positivos, outra para negativos
textos_positivos = df_final[df_final['sentiment'] == 'positive']['review_comment_message'].tolist()
textos_negativos = df_final[df_final['sentiment'] == 'negative']['review_comment_message'].tolist()

def extrair_top_ngrams(corpus, n=10):
    """Função para extrair os n-grams mais comuns de uma lista de textos."""
    # Usamos o CountVectorizer para contar sequências de 2 e 4 palavras
    vec = CountVectorizer(ngram_range=(3, 4), preprocessor=preprocessor).fit(corpus)

    # Soma as contagens de cada n-gram
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)

    # Mapeia as contagens para as palavras/frases
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

# Extrai e mostra os principais motivos para avaliações positivas
top_positivos = extrair_top_ngrams(textos_positivos, n=15)
print("--- Principais Razões (N-grams) para Avaliações POSITIVAS ---")
display(pd.DataFrame(top_positivos, columns=['Frase', 'Frequência']))

# Extrai e mostra os principais motivos para avaliações negativas
top_negativos = extrair_top_ngrams(textos_negativos, n=15)
print("\n--- Principais Razões (N-grams) para Avaliações NEGATIVAS ---")
display(pd.DataFrame(top_negativos, columns=['Frase', 'Frequência']))
- - - - - - - - - - - - - -  OUTPUT - - - - - - - - - - - - - -
  Frase	                  Frequência
0	antes do prazo	        3780
1	chegou antes do	        1150
2	chegou antes do prazo	  917
3	dentro do prazo	        872
4	bem antes do	          818
5	bem antes do prazo	    634
6	entregue antes do	      592
7	entregue antes do prazo	517
8	entrega antes do	      407
9	chegou bem antes	      393
10	produto chegou antes	383
11	de ótima qualidade	  367
12	produto muito bom	    364
13	entregue no prazo	    362
14	chegou bem antes do	  348

  Frase	                    Frequência
0	não recebi produto	      624
1	não foi entregue	        395
2	ainda não recebi	        368
3	produto não foi	          199
4	até agora não	            197
5	prazo de entrega	        186
6	não recebi meu	          182
7	recebi meu produto	      180
8	produto não foi entregue	172
9	ainda não recebi produto	160
10	dinheiro de volta	      144
11	não recebi meu produto	141
12	meu dinheiro de	        130
13	produto não chegou	    128
14	até momento não         123
```
Como podemos ver na lista acima, o maior elogio é receber antes do prazo, com alguns comentários comentando da qualidade do produto. Já nos negativos podemos ver que a maior parte das vezes o produto não foi entrega, mostrando queixas também de exigirem remboolso e o alguns falando do prazo de entrega, mostrando os maiores pontos fortes e fracos, segundo os clientes.

