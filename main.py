from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import nltk
import string
from typing import List, Dict, Any
import pdb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from xgboost import XGBClassifier

# Baixar stopwords se necessário
nltk.download('stopwords', quiet=True)


def initialize_labels_with_kmeans(X_labeled, X_unlabeled, y_eval, k, r):
    """
    Inicializa os rótulos dos dados não rotulados usando KMeans.
    Para simplificar, este exemplo usa sempre o cluster com label == 1
    como "centróide de interesse" (pode ser ajustado conforme a lógica desejada).
    """

    # Converte para array se necessário
    X_labeled_arr = np.array(X_labeled) if not isinstance(
        X_labeled, np.ndarray) else X_labeled
    X_unlabeled_arr = np.array(X_unlabeled) if not isinstance(
        X_unlabeled, np.ndarray) else X_unlabeled

    # Une os dados para rodar KMeans de uma só vez
    X_all = np.vstack((X_labeled_arr, X_unlabeled_arr))

    if isinstance(y_eval, pd.Series) or isinstance(y_eval, pd.DataFrame):
        y_eval = y_eval.values  # Converte para numpy.array

    # Executa KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_all)
    cluster_labels_all = kmeans.labels_

    # Aqui, por simplicidade, consideramos que o cluster "1" é o 'centróide de interesse'
    # (mas isso depende da sua lógica)
    centroid = np.mean(X_all[cluster_labels_all == 1], axis=0)

    # Calcula distâncias de cada amostra NÃO rotulada a esse centróide
    distances = np.linalg.norm(X_unlabeled_arr - centroid, axis=1)
    if distances.size == 0:
        return pd.Series(y_eval)

    # Define quantos exemplos "mais distantes" queremos rotular
    n_distant = int(np.ceil(r * len(X_unlabeled_arr)))
    indices_sorted_desc = np.argsort(distances)[::-1]
    indices_top = indices_sorted_desc[:n_distant]

    # IMPORTANTE: `indices_top` são índices LOCAIS em X_unlabeled_arr,
    # mas precisamos modificar o vetor y na posição correta.
    #
    # Buscamos então todos os índices GLOBAIS que correspondem a -1
    unlabeled_global_indices = np.where(y_eval == -1)[0]

    # Agora, pegamos apenas aqueles que foram selecionados como "mais distantes"
    chosen_indices = unlabeled_global_indices[indices_top]

    # Finalmente, rotulamos esses como 0 (ou outro rótulo desejado)
    y_eval[chosen_indices] = 0

    return pd.Series(y_eval)


def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o CSV e retorna o DataFrame."""
    df = pd.read_csv(caminho)
    # Mapeia o label para binário (1: fake, 0: real)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
    return df


def limpar_texto(texto: str) -> str:
    """Limpa o texto removendo pontuação, convertendo para minúsculas 
    e removendo stopwords em português."""
    texto = ''.join([char for char in texto if char not in string.punctuation])
    texto = texto.lower()
    stop_words = set(stopwords.words('portuguese'))
    texto = ' '.join([word for word in texto.split()
                     if word not in stop_words])
    return texto


def preparar_features_word2vec(df: pd.DataFrame, coluna_texto: str, coluna_label: str,
                               vector_size=100, window=5, min_count=1) -> (np.ndarray, np.ndarray, Word2Vec):
    """
    Prepara as features (X) e labels (y) usando Word2Vec.
    - Treina um modelo Word2Vec com todos os textos da coluna_texto.
    - Cada texto é convertido na média dos vetores de suas palavras.
    """

    # Primeiramente, tokenizamos cada frase (simplesmente via split,
    # pois já removemos pontuação e convertemos para minúsculo).
    tokenized_sentences = [texto.split() for texto in df[coluna_texto]]

    # Treina o modelo Word2Vec
    w2v_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42
    )

    # Gera o embedding médio para cada texto
    X_embeddings = []
    for tokens in tokenized_sentences:
        # Filtra apenas tokens que estão no vocabulário do w2v
        valid_tokens = [w for w in tokens if w in w2v_model.wv]
        if len(valid_tokens) > 0:
            # Calcula a média dos vetores
            mean_vec = np.mean([w2v_model.wv[w] for w in valid_tokens], axis=0)
        else:
            # Se não houver tokens válidos (ou frase vazia), usa vetor de zeros
            mean_vec = np.zeros(vector_size)
        X_embeddings.append(mean_vec)

    X_embeddings = np.array(X_embeddings)
    y = df[coluna_label].values

    return X_embeddings, y, w2v_model


def remover_proporcao_rotulos(y_train, proporcao: float, random_state: int = 42):
    """Remove uma proporção de rótulos do conjunto de treinamento,
    substituindo-os por -1."""
    y_train_mod = y_train.copy()
    n_remover = int(proporcao * len(y_train_mod))
    np.random.seed(random_state)
    indices_remover = np.random.choice(
        y_train_mod.index, n_remover, replace=False)
    y_train_mod.loc[indices_remover] = -1
    return y_train_mod


def remover_um_rotulo(y_train, random_state: int = 42, rotulo: int = 1, prop: float = 1):
    """Remove exatamente uma classe de rótulo do conjunto de treinamento."""
    y_train_mod = y_train.copy()
    np.random.seed(random_state)
    indice_remover = np.random.choice(
        y_train_mod[y_train_mod == rotulo].index,
        int(prop * len(y_train_mod[y_train_mod == rotulo])),
        replace=False
    )
    y_train_mod.loc[indice_remover] = -1
    return y_train_mod


def treinar_e_avaliar_modelo(modelo, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Treina um modelo com os dados fornecidos e avalia no conjunto de teste."""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'accuracy': acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }


class ExperimentoLabelPropagation:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def executar_experimento_proporcoes(self, proporcoes: List[float]) -> pd.DataFrame:
        """Executa o experimento com várias proporções de remoção de rótulos."""
        resultados = []

        for prop in proporcoes:
            df_res = self.executar_experimento_remocao(proporcao=prop)
            resultados.append(df_res)
        return pd.DataFrame(resultados)

    def executar_experimento_um_rotulo(self) -> pd.DataFrame:
        """Executa o experimento removendo exatamente um rótulo."""
        return pd.DataFrame([self.executar_experimento_remocao(um_rotulo=True)])

    def executar_experimento_remocao(self, proporcao: float = None, um_rotulo: bool = False) -> Dict[str, Any]:
        """Executa o experimento para um caso de remoção de rótulos (por proporção ou um único rótulo)."""
        if um_rotulo:
            y_train_mod = remover_um_rotulo(self.y_train)
            prop_info = '1_label'
        else:
            y_train_mod = remover_proporcao_rotulos(
                self.y_train, proporcao=proporcao)
            prop_info = proporcao

        # LabelPropagation (no caso, a inicialização de rótulos via KMeans)
        y_train_propagado = initialize_labels_with_kmeans(
            self.X_train[y_train_mod != -1],
            self.X_train[y_train_mod == -1],
            y_train_mod,
            k=2,
            r=0.75
        )
        mask_labeled_lp = (y_train_propagado != -1)
        X_train_lp = self.X_train[mask_labeled_lp]
        y_train_lp = y_train_propagado[mask_labeled_lp]

        # Modelos com LabelPropagation
        lr_lp = LogisticRegression(max_iter=1000, random_state=42)
        xgb_lp = XGBClassifier(eval_metric='logloss', random_state=42)
        lr_lp_res = treinar_e_avaliar_modelo(
            lr_lp, X_train_lp, y_train_lp, self.X_test, self.y_test)
        xgb_lp_res = treinar_e_avaliar_modelo(
            xgb_lp, X_train_lp, y_train_lp, self.X_test, self.y_test)

        # Cenário sem LabelPropagation (descartar não rotulados)
        mask_labeled = (y_train_mod != -1)
        X_train_no_lp = self.X_train[mask_labeled]
        y_train_no_lp = y_train_mod[mask_labeled]

        lr_no_lp = LogisticRegression(max_iter=1000, random_state=42)
        xgb_no_lp = XGBClassifier(eval_metric='logloss', random_state=42)
        lr_no_lp_res = treinar_e_avaliar_modelo(
            lr_no_lp, X_train_no_lp, y_train_no_lp, self.X_test, self.y_test)
        xgb_no_lp_res = treinar_e_avaliar_modelo(
            xgb_no_lp, X_train_no_lp, y_train_no_lp, self.X_test, self.y_test)

        resultado = {
            'proporcao_remocao': prop_info,
            # Com LabelPropagation (Logistic Regression)
            'acc_log_lp': lr_lp_res['accuracy'],
            'precision_log_lp': lr_lp_res['precision'],
            'recall_log_lp': lr_lp_res['recall'],
            'f1_log_lp': lr_lp_res['f1'],

            # Com LabelPropagation (XGB)
            'acc_xgb_lp': xgb_lp_res['accuracy'],
            'precision_xgb_lp': xgb_lp_res['precision'],
            'recall_xgb_lp': xgb_lp_res['recall'],
            'f1_xgb_lp': xgb_lp_res['f1'],

            # Sem LabelPropagation (Logistic Regression)
            'acc_log_no_lp': lr_no_lp_res['accuracy'],
            'precision_log_no_lp': lr_no_lp_res['precision'],
            'recall_log_no_lp': lr_no_lp_res['recall'],
            'f1_log_no_lp': lr_no_lp_res['f1'],

            # Sem LabelPropagation (XGB)
            'acc_xgb_no_lp': xgb_no_lp_res['accuracy'],
            'precision_xgb_no_lp': xgb_no_lp_res['precision'],
            'recall_xgb_no_lp': xgb_no_lp_res['recall'],
            'f1_xgb_no_lp': xgb_no_lp_res['f1']
        }

        return resultado


def main():
    # Caminho do arquivo CSV pré-processado
    caminho_dados = './dados/pre-processed.csv'

    # Carregar dados
    df = carregar_dados(caminho_dados)

    # Limpar texto
    df['manchete_limpa'] = df['preprocessed_news'].apply(
        lambda txt: limpar_texto(txt))

    # Preparar features usando Word2Vec
    X, y, w2v_model = preparar_features_word2vec(df, 'manchete_limpa', 'label',
                                                 vector_size=100, window=5, min_count=1)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2025
    )

    # Proporções de remoção de rótulos
    proporcoes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    # Executar experimento
    experimento = ExperimentoLabelPropagation(
        X_train, pd.Series(y_train), X_test, y_test)
    df_resultados_prop = experimento.executar_experimento_proporcoes(
        proporcoes)

    # (Opcional) Executar experimento removendo apenas um rótulo
    # df_resultado_um = experimento.executar_experimento_um_rotulo()

    # Concatenar resultados (neste exemplo, só estamos usando df_resultados_prop)
    df_resultados = pd.concat([df_resultados_prop], ignore_index=True)

    # Salvar resultados
    df_resultados.to_csv(
        'comparacao_labelpropagation_vs_sem_word2vec.csv', index=False)
    print("Resultados salvos em 'comparacao_labelpropagation_vs_sem_word2vec.csv'.")


if __name__ == '__main__':
    main()
