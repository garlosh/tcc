from common import mcls, datahandler
import pandas as pd
import numpy as np
import nltk
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from dotenv import load_dotenv
import os
load_dotenv(override=True)
nltk.download('stopwords', quiet=True)
# openai.api_key = openai.api_key = os.getenv("OPENAI_API_KEY")


def treinar_e_avaliar_modelo(modelo, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Treina o modelo e avalia-o no conjunto de teste, retornando
    métricas de acurácia, precisão, revocação e F1 (Zhang et al., 2019).
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)
    return {
        'accuracy': acc,
        'precision': rep['weighted avg']['precision'],
        'recall': rep['weighted avg']['recall'],
        'f1': rep['weighted avg']['f1-score']
    }


def executar_experimento(X_train, y_train_ser, X_test, y_test,
                         text_train, text_test,
                         proporcao: float = None, um_rotulo=False) -> Dict[str, Any]:
    """
    Remove rótulos e executa:
      - Cenário sem LP (treina LR, XGB)
      - Cenário com LP (treina LR, XGB)
    Retorna um dicionário com as principais métricas de avaliação.

    Adicionalmente, para cada proporção de rótulos removidos, salva os
    conjuntos de dados de treino e teste em arquivos .csv específicos
    (Silva e Sato, 2021).    
    """
    # 1) Remover rótulos
    y_mod = datahandler.remover_proporcao_rotulos(y_train_ser, proporcao)
    prop_info = proporcao

    # Cria a pasta baseada na proporção de rótulos removidos
    folder_name = f"./dados_gpt/proporcao_{prop_info}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 2) SEM Label Propagation
    mask_no_lp = (y_mod != -1)
    X_no_lp = X_train[mask_no_lp]
    y_no_lp = y_mod[mask_no_lp]
    text_no_lp = [text_train[i]
                  for i in range(len(text_train)) if mask_no_lp[i]]

    # Modelos clássicos
    lr_no_lp = LogisticRegression(max_iter=1000, random_state=42)
    xgb_no_lp = XGBClassifier(eval_metric='logloss', random_state=42)

    lr_no_lp_res = treinar_e_avaliar_modelo(
        lr_no_lp, X_no_lp, y_no_lp, X_test, y_test)
    xgb_no_lp_res = treinar_e_avaliar_modelo(
        xgb_no_lp, X_no_lp, y_no_lp, X_test, y_test)

    # 3) COM Label Propagation (via KMeans)
    y_propagado = mcls.initialize_labels_with_kmeans(
        X_no_lp, X_train[~mask_no_lp], y_mod, k=2, r=0.15
    )
    mask_lp = (y_propagado != -1)
    X_lp = X_train[mask_lp]
    y_lp = y_propagado[mask_lp]

    # Para salvar também os textos usados em LP
    text_lp = [text_train[i] for i in range(len(text_train)) if mask_lp[i]]

    lr_lp = LogisticRegression(max_iter=1000, random_state=42)
    xgb_lp = XGBClassifier(eval_metric='logloss', random_state=42)

    lr_lp_res = treinar_e_avaliar_modelo(lr_lp, X_lp, y_lp, X_test, y_test)
    xgb_lp_res = treinar_e_avaliar_modelo(xgb_lp, X_lp, y_lp, X_test, y_test)

    # ========== SALVAR DADOS EM CSV ==========

    # Cria DataFrame de treinamento SEM LP
    df_train_no_lp = pd.DataFrame({'text': text_no_lp, 'label': y_no_lp})
    df_train_no_lp.to_csv(os.path.join(
        folder_name, 'train_no_lp.csv'), index=False)

    # Cria DataFrame de teste SEM LP
    df_test_no_lp = pd.DataFrame({'text': text_test, 'label': y_test})
    df_test_no_lp.to_csv(os.path.join(
        folder_name, 'test.csv'), index=False)

    # Cria DataFrame de treinamento COM LP
    df_train_lp = pd.DataFrame({'text': text_lp, 'label': y_lp})
    df_train_lp.to_csv(os.path.join(folder_name, 'train_lp.csv'), index=False)

    # 4) Monta dicionário final com as métricas
    return {
        'proporcao_remocao': prop_info,

        # SEM LP (LR, XGB)
        'acc_log_no_lp': lr_no_lp_res['accuracy'],
        'precision_log_no_lp': lr_no_lp_res['precision'],
        'recall_log_no_lp': lr_no_lp_res['recall'],
        'f1_log_no_lp': lr_no_lp_res['f1'],

        'acc_xgb_no_lp': xgb_no_lp_res['accuracy'],
        'precision_xgb_no_lp': xgb_no_lp_res['precision'],
        'recall_xgb_no_lp': xgb_no_lp_res['recall'],
        'f1_xgb_no_lp': xgb_no_lp_res['f1'],

        # COM LP (LR, XGB)
        'acc_log_lp': lr_lp_res['accuracy'],
        'precision_log_lp': lr_lp_res['precision'],
        'recall_log_lp': lr_lp_res['recall'],
        'f1_log_lp': lr_lp_res['f1'],

        'acc_xgb_lp': xgb_lp_res['accuracy'],
        'precision_xgb_lp': xgb_lp_res['precision'],
        'recall_xgb_lp': xgb_lp_res['recall'],
        'f1_xgb_lp': xgb_lp_res['f1'],
    }


def main():
    """
    Função principal que carrega os dados, realiza pré-processamento,
    gera embeddings e executa a rotina de experimentos para diferentes
    proporções de rótulos removidos (Zhang et al., 2019).
    """
    # Carregamento e pré-processamento
    caminho_dados = './dados/pre-processed.csv'
    caminho_arquivo = 'resultados_tcc.csv'
    df = datahandler.carregar_dados(caminho_dados)
    df['manchete_ia'] = df['preprocessed_news']
    df['manchete_limpa'] = df['preprocessed_news'].apply(
        datahandler.limpar_texto)

    # Gera embeddings
    X, y, _ = datahandler.preparar_features_word2vec(df, 'manchete_limpa', 'label',
                                                     vector_size=100, window=5, min_count=1)
    textos = df['manchete_ia'].values

    # Divide treino/teste
    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, textos, test_size=0.2, random_state=2025
    )

    # Convertendo y_train para Series (facilita indexar e remover rótulos)
    y_train_ser = pd.Series(y_train)

    # Lista de proporções de rótulos que serão removidos
    proporcoes = [0.0, 0.50, 0.75]  # Exemplo

    # Executa experimentos
    resultados = []

    for prop in proporcoes:
        # Executa o experimento, salvando resultados e arquivos de dados .csv
        res_dict = executar_experimento(
            X_train, y_train_ser, X_test, y_test,
            text_train, text_test,
            proporcao=prop
        )

        resultados.append(res_dict)
        df_resultados = pd.DataFrame(resultados)

        file_exists = os.path.isfile(caminho_arquivo)
        df_resultados.to_csv(
            caminho_arquivo,
            mode='a',
            index=False,
            header=not file_exists  # Adiciona o cabeçalho somente se o arquivo não existir
        )

        # Limpa a lista de resultados para evitar duplicações no CSV final
        resultados.clear()

        print(f"Resultados intermediários com proporção {
              prop} salvos em '{caminho_arquivo}'.")


if __name__ == '__main__':
    main()
