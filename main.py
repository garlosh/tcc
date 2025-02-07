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


def executar_experimento(df, proporcao: float = None, seeds: list = None) -> pd.DataFrame:
    if seeds is None:
        raise ValueError(
            "Uma lista de seeds deve ser fornecida para garantir a reprodutibilidade.")

    metrics_list = []

    for seed in seeds:
        np.random.seed(seed)

        # Divisão de treino e teste com seed
        X, y, _ = datahandler.preparar_features_word2vec(
            df, 'manchete_limpa', 'label', vector_size=100, window=5, min_count=1)
        textos = df['manchete_ia'].values

        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
            X, y, textos, test_size=0.2, random_state=seed)
        y_train_ser = pd.Series(y_train)

        y_mod = datahandler.remover_proporcao_rotulos(y_train_ser, proporcao)

        # Criar a pasta baseada na proporção de rótulos removidos e seed
        folder_name = f"./dados_gpt/proporcao_{proporcao}/seed_{seed}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # SEM Label Propagation
        mask_no_lp = (y_mod != -1)
        X_no_lp = X_train[mask_no_lp]
        y_no_lp = y_mod[mask_no_lp]

        lr_no_lp = LogisticRegression(max_iter=1000, random_state=seed)
        xgb_no_lp = XGBClassifier(eval_metric='logloss', random_state=seed)

        lr_no_lp_res = treinar_e_avaliar_modelo(
            lr_no_lp, X_no_lp, y_no_lp, X_test, y_test)
        xgb_no_lp_res = treinar_e_avaliar_modelo(
            xgb_no_lp, X_no_lp, y_no_lp, X_test, y_test)

        # COM Label Propagation
        y_propagado = mcls.initialize_labels_with_kmeans(
            X_no_lp, X_train[~mask_no_lp], y_mod, k=2, r=0.15, random_state=seed)
        mask_lp = (y_propagado != -1)
        X_lp = X_train[mask_lp]
        y_lp = y_propagado[mask_lp]

        lr_lp = LogisticRegression(max_iter=1000, random_state=seed)
        xgb_lp = XGBClassifier(eval_metric='logloss', random_state=seed)

        lr_lp_res = treinar_e_avaliar_modelo(lr_lp, X_lp, y_lp, X_test, y_test)
        xgb_lp_res = treinar_e_avaliar_modelo(
            xgb_lp, X_lp, y_lp, X_test, y_test)

        # Salvar os dados em CSV
        df_train_no_lp = pd.DataFrame({'text': [text_train[i] for i in range(
            len(text_train)) if mask_no_lp[i]], 'label': y_no_lp})
        df_train_no_lp.to_csv(os.path.join(
            folder_name, 'train_no_lp.csv'), index=False)

        df_test_no_lp = pd.DataFrame({'text': text_test})
        df_test_no_lp.to_csv(os.path.join(
            folder_name, 'test.csv'), index=False)

        df_train_lp = pd.DataFrame({'text': [text_train[i] for i in range(
            len(text_train)) if mask_lp[i]], 'label': y_lp})
        df_train_lp.to_csv(os.path.join(
            folder_name, 'train_lp.csv'), index=False)

        # Adiciona resultados para esta execução
        metrics_list.append({
            'proporcao_remocao': proporcao,
            'seed': seed,
            'accuracy_log_no_lp': lr_no_lp_res['accuracy'],
            'precision_log_no_lp': lr_no_lp_res['precision'],
            'recall_log_no_lp': lr_no_lp_res['recall'],
            'f1_log_no_lp': lr_no_lp_res['f1'],

            'accuracy_xgb_no_lp': xgb_no_lp_res['accuracy'],
            'precision_xgb_no_lp': xgb_no_lp_res['precision'],
            'recall_xgb_no_lp': xgb_no_lp_res['recall'],
            'f1_xgb_no_lp': xgb_no_lp_res['f1'],

            'accuracy_log_lp': lr_lp_res['accuracy'],
            'precision_log_lp': lr_lp_res['precision'],
            'recall_log_lp': lr_lp_res['recall'],
            'f1_log_lp': lr_lp_res['f1'],

            'accuracy_xgb_lp': xgb_lp_res['accuracy'],
            'precision_xgb_lp': xgb_lp_res['precision'],
            'recall_xgb_lp': xgb_lp_res['recall'],
            'f1_xgb_lp': xgb_lp_res['f1'],
        })

    # Converte os resultados em DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Retorna DataFrame com médias e desvios padrão separados por métrica
    summary = metrics_df.groupby('proporcao_remocao').agg({
        'accuracy_log_no_lp': ['mean', 'std'],
        'precision_log_no_lp': ['mean', 'std'],
        'recall_log_no_lp': ['mean', 'std'],
        'f1_log_no_lp': ['mean', 'std'],
        'accuracy_xgb_no_lp': ['mean', 'std'],
        'precision_xgb_no_lp': ['mean', 'std'],
        'recall_xgb_no_lp': ['mean', 'std'],
        'f1_xgb_no_lp': ['mean', 'std'],
        'accuracy_log_lp': ['mean', 'std'],
        'precision_log_lp': ['mean', 'std'],
        'recall_log_lp': ['mean', 'std'],
        'f1_log_lp': ['mean', 'std'],
        'accuracy_xgb_lp': ['mean', 'std'],
        'precision_xgb_lp': ['mean', 'std'],
        'recall_xgb_lp': ['mean', 'std'],
        'f1_xgb_lp': ['mean', 'std']
    }).reset_index()

    summary.columns = ['_'.join(col).strip('_')
                       for col in summary.columns.values]

    return summary


def main():
    caminho_dados = './dados/pre-processed.csv'
    caminho_arquivo = 'resultados_tcc.csv'
    df = datahandler.carregar_dados(caminho_dados)
    df['manchete_ia'] = df['preprocessed_news']
    df['manchete_limpa'] = df['preprocessed_news'].apply(
        datahandler.limpar_texto)

    proporcoes = [0.0, 0.50, 0.75]
    seeds = [42, 43, 44, 45, 46]  # Lista de seeds para reprodutibilidade
    resultados = []

    for prop in proporcoes:
        df_resultados = executar_experimento(df, proporcao=prop, seeds=seeds)
        resultados.append(df_resultados)

    # Combina todos os resultados em um único DataFrame e salva
    final_resultados = pd.concat(resultados, ignore_index=True)
    final_resultados.to_csv(caminho_arquivo, index=False)

    print(f"Resultados salvos em '{caminho_arquivo}'.")


if __name__ == '__main__':
    main()
