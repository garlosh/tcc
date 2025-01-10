from typing import List, Dict
import openai
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def build_few_shot_examples(text_train: List[str],
                            y_train,  # pode ser array ou Series
                            max_por_classe=5):
    exemplos = []
    fake_count = real_count = 0

    # iteramos diretamente em pares
    for texto, label in zip(text_train, y_train):
        if label == 1 and fake_count < max_por_classe:
            exemplos.append({"texto": texto, "rotulo": "fake"})
            fake_count += 1
        elif label == 0 and real_count < max_por_classe:
            exemplos.append({"texto": texto, "rotulo": "real"})
            real_count += 1

        if fake_count >= max_por_classe and real_count >= max_por_classe:
            break

    return exemplos


def prever_com_chatgpt(texto: str, exemplos_rotulados: List[Dict[str, str]] = None) -> str:
    """
    Faz uma previsão usando ChatGPT:
      - Se exemplos_rotulados for None -> Zero-Shot
      - Caso contrário -> Few-Shot
    Retorna 'fake' ou 'real'.
    """
    # Ajuste sua chave de API

    if exemplos_rotulados is None:
        # Zero-shot
        prompt = (
            "Você é um modelo de IA que classifica notícias como 'fake' ou 'real'.\n"
            "Responda apenas 'fake' ou 'real'.\n\n"
            f"Notícia: {texto}\n"
            "Classifique: "
        )
    else:
        # Few-shot
        exemplos_str = ""
        for ex in exemplos_rotulados:
            exemplos_str += f"Exemplo:\nNotícia: {
                ex['texto']}\nRótulo: {ex['rotulo']}\n\n"

        prompt = (
            "Você é um modelo de IA que classifica notícias como 'fake' ou 'real'.\n"
            "Responda apenas 'fake' ou 'real'.\n\n"
            + exemplos_str +
            f"Agora, classifique a seguinte notícia:\nNotícia: {
                texto}\nClassifique: "
        )

    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um classificador de fake news."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2,
            temperature=0.0
        )
        conteudo = resposta['choices'][0]['message']['content'].strip().lower()
        return 'fake' if 'fake' in conteudo else 'real'
    except Exception as e:
        print("Erro ao chamar ChatGPT:", e)
        return 'real'


def avaliar_chatgpt(textos: List[str], y_true: np.ndarray, exemplos_rotulados: List[Dict[str, str]] = None
                    ) -> Dict[str, float]:
    """
    Avalia ChatGPT em zero-shot (exemplos_rotulados=None) ou few-shot (exemplos_rotulados != None).
    """
    y_pred = []
    for txt in textos:
        pred = prever_com_chatgpt(txt, exemplos_rotulados)
        y_pred.append(1 if pred == 'fake' else 0)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'accuracy': acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }
