import pandas as pd
from typing import Tuple
import string
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)


def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
    return df


def limpar_texto(texto: str) -> str:
    texto = ''.join([c for c in texto if c not in string.punctuation])
    texto = texto.lower()
    stop_words = set(stopwords.words('portuguese'))
    return ' '.join([w for w in texto.split() if w not in stop_words])


def preparar_features_word2vec(df: pd.DataFrame, col_texto: str, col_label: str,
                               vector_size=100, window=5, min_count=1):
    """
    Gera embeddings via Word2Vec (média dos vetores das palavras).
    Retorna X, y, e o modelo W2V treinado.
    """
    # Tokenização simples
    tokenized = [txt.split() for txt in df[col_texto]]

    w2v_model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42
    )

    X_emb = []
    for tokens in tokenized:
        valid = [w for w in tokens if w in w2v_model.wv]
        if valid:
            X_emb.append(np.mean([w2v_model.wv[w] for w in valid], axis=0))
        else:
            X_emb.append(np.zeros(vector_size))

    y = df[col_label].values
    return np.array(X_emb), y, w2v_model


def remover_proporcao_rotulos(y_train: pd.Series, prop: float, random_state=42):
    """
    Coloca -1 em 'prop' proporção dos rótulos.
    """
    y_mod = y_train.copy()
    n_remover = int(prop * len(y_mod))
    np.random.seed(random_state)
    idx_remover = np.random.choice(y_mod.index, n_remover, replace=False)
    y_mod.loc[idx_remover] = -1
    return y_mod
