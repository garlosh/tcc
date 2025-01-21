import openai  # pip install openai
from common import gpt, mcls, datahandler
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
from time import gmtime, strftime
import hashlib
load_dotenv(override=True)
nltk.download('stopwords', quiet=True)
# openai.api_key = openai.api_key = os.getenv("OPENAI_API_KEY")


def remover_um_rotulo(y_train: pd.Series, rotulo=1, prop=1.0, random_state=42):
    """
    Remove completamente uma classe (ou parte dela).
    """
    y_mod = y_train.copy()
    np.random.seed(random_state)
    idx = y_mod[y_mod == rotulo].index
    n_remover = int(prop * len(idx))
    idx_remover = np.random.choice(idx, n_remover, replace=False)
    y_mod.loc[idx_remover] = -1
    return y_mod


def treinar_e_avaliar_modelo(modelo, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
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
      - Cenário sem LP (treina LR, XGB + ChatGPT zero/few-shot)
      - Cenário com LP (treina LR, XGB + ChatGPT zero/few-shot)
    Retorna um dicionário com as métricas.
    """
    # 1) Remover rótulos
    y_mod = datahandler.remover_proporcao_rotulos(y_train_ser, proporcao)
    prop_info = proporcao

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

    # ChatGPT (Zero-Shot e Few-Shot)

    chatgpt_zero_no_lp = gpt.avaliar_chatgpt(
        text_no_lp, y_no_lp)  # zero-shot
    # exemplos_fs_no_lp = gpt.build_few_shot_examples(text_no_lp, y_no_lp)
    # chatgpt_few_no_lp = gpt.avaliar_chatgpt(
    #    text_no_lp, y_no_lp, exemplos_fs_no_lp)

    # 3) COM Label Propagation (via KMeans)
    y_propagado = mcls.initialize_labels_with_kmeans(
        X_no_lp, X_train[~mask_no_lp], y_mod, k=2, r=0.15
    )
    mask_lp = (y_propagado != -1)
    X_lp = X_train[mask_lp]
    y_lp = y_propagado[mask_lp]
    text_lp = [text_train[i] for i in range(len(text_train)) if mask_lp[i]]

    # Modelos clássicos
    lr_lp = LogisticRegression(max_iter=1000, random_state=42)
    xgb_lp = XGBClassifier(eval_metric='logloss', random_state=42)

    lr_lp_res = treinar_e_avaliar_modelo(lr_lp, X_lp, y_lp, X_test, y_test)
    xgb_lp_res = treinar_e_avaliar_modelo(xgb_lp, X_lp, y_lp, X_test, y_test)

    # ChatGPT (Zero-Shot e Few-Shot)
    chatgpt_zero_lp = gpt.avaliar_chatgpt(
        text_lp, y_lp)  # zero-shot
    # exemplos_fs_lp = gpt.build_few_shot_examples(text_lp, y_lp)
    # chatgpt_few_lp = gpt.avaliar_chatgpt(
    #    text_lp, y_lp, exemplos_fs_lp)

    # 4) Monta dicionário final
    return {
        'proporcao_remocao': prop_info,

        # SEM LP (LR, XGB, ChatGPT Zero, ChatGPT Few)
        'acc_log_no_lp': lr_no_lp_res['accuracy'],
        'precision_log_no_lp': lr_no_lp_res['precision'],
        'recall_log_no_lp': lr_no_lp_res['recall'],
        'f1_log_no_lp': lr_no_lp_res['f1'],

        'acc_xgb_no_lp': xgb_no_lp_res['accuracy'],
        'precision_xgb_no_lp': xgb_no_lp_res['precision'],
        'recall_xgb_no_lp': xgb_no_lp_res['recall'],
        'f1_xgb_no_lp': xgb_no_lp_res['f1'],

        'acc_chatgpt_zero_no_lp': chatgpt_zero_no_lp['accuracy'],
        'prec_chatgpt_zero_no_lp': chatgpt_zero_no_lp['precision'],
        'rec_chatgpt_zero_no_lp': chatgpt_zero_no_lp['recall'],
        'f1_chatgpt_zero_no_lp': chatgpt_zero_no_lp['f1'],

        # COM LP (LR, XGB, ChatGPT Zero, ChatGPT Few)
        'acc_log_lp': lr_lp_res['accuracy'],
        'precision_log_lp': lr_lp_res['precision'],
        'recall_log_lp': lr_lp_res['recall'],
        'f1_log_lp': lr_lp_res['f1'],

        'acc_xgb_lp': xgb_lp_res['accuracy'],
        'precision_xgb_lp': xgb_lp_res['precision'],
        'recall_xgb_lp': xgb_lp_res['recall'],
        'f1_xgb_lp': xgb_lp_res['f1'],

        'acc_chatgpt_zero_lp': chatgpt_zero_lp['accuracy'],
        'prec_chatgpt_zero_lp': chatgpt_zero_lp['precision'],
        'rec_chatgpt_zero_lp': chatgpt_zero_lp['recall'],
        'f1_chatgpt_zero_lp': chatgpt_zero_lp['f1']
    }


def main():
    # Carregamento e pré-processamento
    caminho_dados = './dados/pre-processed.csv'
    caminho_arquivo = 'resultados_tcc.csv'
    df['manchete_ia'] = df['preprocessed_news']
    df = datahandler.carregar_dados(caminho_dados)
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

    # Lista de proporções
    proporcoes = [0.0, 0.25, 0.50, 0.75, 0.85]  # por exemplo

    # Executa experimentos

    # Inicializa a lista de resultados
    resultados = []

    # Itera pelas proporções
    for prop in proporcoes:
        # Executa o experimento
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
        resultados.clear()
        # Log para indicar que os resultados foram salvos
        print(f"Resultados intermediários com proporção {
              prop} salvos em '{caminho_arquivo}'.")


if __name__ == '__main__':
    main()
