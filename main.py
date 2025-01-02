from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import nltk
import string
from typing import List, Dict, Any
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from xgboost import XGBClassifier

# Baixar stopwords se necessário
nltk.download('stopwords', quiet=True)


def initialize_labels_with_kmeans(X_labeled, X_unlabeled, k, r):
    """
    Inicializa os rótulos dos dados não rotulados usando k-means, conforme descrito.

    Parâmetros
    ----------
    X_labeled : csr_matrix ou np.array de shape (n_labeled, n_features)
        Dados com rótulos conhecidos (treinamento rotulado).
    y_labeled : np.array de shape (n_labeled,)
        Rótulos conhecidos (0/1 ou -1/+1).
    X_unlabeled : csr_matrix ou np.array de shape (n_unlabeled, n_features)
        Dados não rotulados.
    k : int
        Número de clusters para o k-means.
    r : float ou int
        Parâmetro que controla quantos exemplos negativos serão selecionados
        dos clusters definidos como negativos:
          - Se for float (ex.: 0.2), seleciona essa fração do total de exemplos não rotulados de cada cluster negativo.
          - Se for int (ex.: 10), seleciona esse número fixo de exemplos de cada cluster negativo.

    Retorna
    -------
    y_unlabeled_pred : np.array de shape (n_unlabeled,)
        Rótulos previstos para os dados não rotulados, após o processo de
        definição via k-means.
    """

    X_labeled_arr = X_labeled.toarray() if hasattr(
        X_labeled, 'toarray') else X_labeled
    X_unlabeled_arr = X_unlabeled.toarray() if hasattr(
        X_unlabeled, 'toarray') else X_unlabeled
    X_all = np.vstack((X_labeled_arr, X_unlabeled_arr))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_all)

    cluster_labels_all = kmeans.labels_

    centroid = np.mean(X_all[cluster_labels_all == 1], axis=0)
    distances = np.linalg.norm(X_all - centroid, axis=1)
    n_distant = int(np.ceil(r * len(X_all)))
    indices_sorted_desc = np.argsort(distances)[::-1]
    indices_top = indices_sorted_desc[:n_distant]
    y_weak_labeled = np.full(len(X_all), -1)
    y_weak_labeled[indices_top] = 0
    return y_weak_labeled


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


def preparar_features(df: pd.DataFrame, coluna_texto: str, coluna_label: str):
    """Prepara as features (X) e labels (y) usando TF-IDF."""
    tfidf = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
    X = tfidf.fit_transform(df[coluna_texto],)
    y = df[coluna_label]
    return X, y, tfidf


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
    indice_remover = np.random.choice(y_train_mod[y_train_mod == rotulo].index,
                                      int(.99 * len(y_train_mod[y_train_mod == rotulo])), replace=False)
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
            # pdb.set_trace()
            prop_info = '1_label'
        else:
            y_train_mod = remover_proporcao_rotulos(
                self.y_train, proporcao=proporcao)
            prop_info = proporcao

        # LabelPropagation
        y_train_propagado = initialize_labels_with_kmeans(
            self.X_train[y_train_mod != -1], self.X_train[y_train_mod == -1], k=2, r=.75)

        # Modelos com LabelPropagation
        lr_lp = LogisticRegression(max_iter=1000, random_state=42)
        xgb_lp = XGBClassifier(eval_metric='logloss',
                               use_label_encoder=False, random_state=42)
        lr_lp_res = treinar_e_avaliar_modelo(
            lr_lp, self.X_train, y_train_propagado, self.X_test, self.y_test)
        xgb_lp_res = treinar_e_avaliar_modelo(
            xgb_lp, self.X_train, y_train_propagado, self.X_test, self.y_test)

        # Cenário sem LabelPropagation (descartar não rotulados)
        mask_labeled = (y_train_mod != -1)
        X_train_no_lp = self.X_train[mask_labeled]
        y_train_no_lp = y_train_mod[mask_labeled]

        lr_no_lp = LogisticRegression(max_iter=1000, random_state=42)
        xgb_no_lp = XGBClassifier(
            eval_metric='logloss', use_label_encoder=False, random_state=42)
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

    # Preparar features
    X, y, _ = preparar_features(df, 'manchete_limpa', 'label')

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Proporções de remoção de rótulos
    proporcoes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    # Executar experimento
    experimento = ExperimentoLabelPropagation(X_train, y_train, X_test, y_test)
    df_resultados_prop = experimento.executar_experimento_proporcoes(
        proporcoes)

    # Executar experimento removendo apenas um rótulo

    df_resultado_um = experimento.executar_experimento_um_rotulo()

    # Concatenar resultados
    df_resultados = pd.concat(
        [df_resultados_prop, df_resultado_um], ignore_index=True)

    # Salvar resultados
    df_resultados.to_csv('comparacao_labelpropagation_vs_sem.csv', index=False)
    print("Resultados salvos em 'comparacao_labelpropagation_vs_sem.csv'.")


if __name__ == '__main__':
    main()
