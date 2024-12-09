import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.semi_supervised import LabelPropagation
from nltk.corpus import stopwords
from xgboost import XGBClassifier
import string
import nltk
import numpy as np

nltk.download('stopwords')

# Carregar o CSV
df = pd.read_csv('./dados/pre-processed.csv')

# Converter os rótulos para valores numéricos
df['label'] = df['label'].apply(lambda x: 1 if x == 'fake' else 0)

# Função para limpar as manchetes (remoção de pontuação e stopwords)
def limpar_texto(texto):
    # Remover pontuações
    texto = ''.join([char for char in texto if char not in string.punctuation])
    # Converter para minúsculas
    texto = texto.lower()
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    texto = ' '.join([word for word in texto.split() if word not in stop_words])
    return texto

# Aplicar a função de limpeza nas manchetes
df['manchete_limpa'] = df['preprocessed_news'].apply(limpar_texto)

# Inicializar o vetorizador TF-IDF
tfidf = TfidfVectorizer()

# Aplicar o vetorizador nas manchetes limpas
X_tfidf = tfidf.fit_transform(df['manchete_limpa'])

# Separar a variável preditora (X) e a variável resposta (y)
X = X_tfidf  # A matriz TF-IDF é a nossa variável preditora
y = df['label']  # Coluna 'label' como variável resposta

# Dividir o dataset em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, df.index, test_size=0.2, random_state=42)

# Definir diferentes níveis de omissão de rótulos
proporcoes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# DataFrame para armazenar resultados
resultados = []

for prop in proporcoes:
    # Copiar y_train
    y_train_semi = y_train.copy()
    
    # Determinar quantos rótulos serão removidos
    n_remover = int(prop * len(y_train_semi))
    indices_remover = np.random.choice(y_train_semi.index, n_remover, replace=False)
    
    # Remover (ocultar) esses rótulos configurando-os como -1
    y_train_semi.loc[indices_remover] = -1
    
    # Aplicar LabelPropagation
    lp = LabelPropagation()
    lp.fit(X_train, y_train_semi)
    y_train_propagado = lp.transduction_
    
    # Treinar Logistic Regression com os rótulos propagados
    modelo_log = LogisticRegression(max_iter=1000)
    modelo_log.fit(X_train, y_train_propagado)
    y_pred_log = modelo_log.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)
    report_log = classification_report(y_test, y_pred_log, output_dict=True)
    
    # Treinar XGBoost com os rótulos propagados
    modelo_xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    modelo_xgb.fit(X_train, y_train_propagado)
    y_pred_xgb = modelo_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
    
    # Armazenar resultados desta proporção
    resultados.append({
        'proporcao_remocao': prop,
        'acc_log': acc_log,
        'precision_log': report_log['weighted avg']['precision'],
        'recall_log': report_log['weighted avg']['recall'],
        'f1_log': report_log['weighted avg']['f1-score'],
        'acc_xgb': acc_xgb,
        'precision_xgb': report_xgb['weighted avg']['precision'],
        'recall_xgb': report_xgb['weighted avg']['recall'],
        'f1_xgb': report_xgb['weighted avg']['f1-score']
    })

# Converter resultados para DataFrame
df_resultados = pd.DataFrame(resultados)

# Salvar os resultados finais em CSV
df_resultados.to_csv('comparacao_labelpropagation.csv', index=False)
print("Resultados salvos em 'comparacao_labelpropagation.csv'.")
