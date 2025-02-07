import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pdb


def carregar_dados(arquivo_classificado):
    dados_classificados = pd.read_csv(arquivo_classificado)
    y_pred = dados_classificados['label'].tolist()
    return y_pred


def carregar_rotulos_originais(arquivo_teste, dados_originais):
    dados_teste = pd.read_csv(arquivo_teste)
    dados_teste = dados_teste.merge(
        dados_originais, left_on='text', right_on='preprocessed_news', how='left')
    return dados_teste['label'].apply(lambda x: 1 if x == 'fake' else 0).tolist()


def calcular_metricas(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True)
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': rep['weighted avg']['precision'],
        'Recall': rep['weighted avg']['recall'],
        'F1-Score': rep['weighted avg']['f1-score']
    }


def processar_resultados():
    proporcoes = [0.0, 0.50, 0.75]
    seeds = [42, 43, 44, 45, 46]  # Lista de seeds utilizadas
    tipos = ['lp', 'no_lp']
    resultados = []

    # Carregar os dados originais
    dados_originais = pd.read_csv(
        './dados/pre-processed.csv').drop_duplicates(subset=['preprocessed_news'])

    for prop in proporcoes:
        for t in tipos:
            metricas_seeds = []

            for seed in seeds:
                arquivo_teste = f"./dados_gpt/proporcao_{
                    prop}/seed_{seed}/test.csv"
                arquivo_classificado = f"./dados_gpt/proporcao_{
                    prop}/seed_{seed}/resultado/resultado_{t}.csv"

                if os.path.exists(arquivo_teste) and os.path.exists(arquivo_classificado):
                    y_true = carregar_rotulos_originais(
                        arquivo_teste, dados_originais)
                    y_pred = carregar_dados(arquivo_classificado)
                    metricas = calcular_metricas(y_true, y_pred)
                    metricas_seeds.append(metricas)
            pdb.set_trace()
            if metricas_seeds:
                df_metricas = pd.DataFrame(metricas_seeds)
                media_metricas = df_metricas.mean().to_dict()
                desvio_metricas = df_metricas.std().to_dict()

                resultados.append({
                    'proporcao_remocao': prop,
                    f'acc_gpt_{t}_mean': media_metricas['Acurácia'],
                    f'precision_gpt_{t}_mean': media_metricas['Precisão'],
                    f'recall_gpt_{t}_mean': media_metricas['Recall'],
                    f'f1_gpt_{t}_mean': media_metricas['F1-Score'],
                    f'acc_gpt_{t}_std': desvio_metricas['Acurácia'],
                    f'precision_gpt_{t}_std': desvio_metricas['Precisão'],
                    f'recall_gpt_{t}_std': desvio_metricas['Recall'],
                    f'f1_gpt_{t}_std': desvio_metricas['F1-Score']
                })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv('resultados_tcc_gpt.csv', index=False)
    print("Resultados salvos em 'resultados_tcc_gpt.csv'")


if __name__ == "__main__":
    processar_resultados()
