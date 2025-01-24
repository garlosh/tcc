import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pdb


def carregar_dados(arquivo_teste, arquivo_classificado):
    """
    Carrega os dados de teste originais e os dados classificados do modelo.

    Parâmetros:
        arquivo_teste (str): Caminho para o arquivo CSV com os dados de teste originais.
        arquivo_classificado (str): Caminho para o arquivo CSV com os dados classificados pelo modelo.

    Retorna:
        y_true (list): Valores reais (originais) das classes.
        y_pred (list): Valores previstos pelo modelo.
    """
    # Carregar dados originais de teste
    dados_teste = pd.read_csv(arquivo_teste)

    # Carregar dados classificados pelo modelo
    dados_classificados = pd.read_csv(arquivo_classificado)

    # Garantir que as colunas tenham os mesmos nomes para unificar o processo
    # Supondo que a coluna com os rótulos seja 'label' em ambos os arquivos
    y_true = dados_teste['label'].tolist()
    y_pred = dados_classificados['label'].tolist()

    return y_true, y_pred


def calcular_metricas(y_true, y_pred):
    """
    Calcula métricas de avaliação com base nos valores reais e previstos.

    Parâmetros:
        y_true (list): Valores reais (originais) das classes.
        y_pred (list): Valores previstos pelo modelo.

    Retorna:
        dict: Dicionário contendo as métricas de avaliação.
    """
    metrics = {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted')
    }

    return metrics


def main():
    arquivo_teste = './dados_gpt/test_original.csv'

    # Lista de proporções de rótulos que serão removidos
    proporcoes = [0.0, 0.50, 0.75]  # Exemplo
    resultados = []
    tipos = ['lp', 'no_lp']
    for t in tipos:
        for prop in proporcoes:
            arquivo_classificado = f'''./dados_gpt/proporcao_{
                prop}/resultado/test_predictions_{t}.csv'''
            # Carregar dados
            y_true, y_pred = carregar_dados(
                arquivo_teste, arquivo_classificado)
            metricas = calcular_metricas(y_true, y_pred)
            resultados.append({'proporcao_remocao': prop,
                               f'acc_gpt_{t}': metricas['Acurácia'],
                               f'precision_gpt_{t}': metricas['Precisão'],
                               f'recall_gpt_{t}': metricas['Recall'],
                               f'f1_gpt_{t}': metricas['F1-Score']})

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(
        'resultados_tcc_gpt.csv',
        index=False,
    )


if __name__ == "__main__":
    main()
