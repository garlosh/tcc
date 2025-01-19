from typing import List, Dict
import openai
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import json
import requests
import pdb


def build_few_shot_examples(text_train: List[str],
                            y_train,  # pode ser array ou Series
                            max_por_classe=2):
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
    # pdb.set_trace()

    return exemplos


def prever_com_chatgpt(texto: str, exemplos_rotulados: List[Dict[str, str]] = None,
                       url_ollama: str = "http://localhost:11434/api/generate", modelo_ollama: str = "qwen2.5:7b") -> str:
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
            "Seu objetivo é classificar notícias como 'fake' ou 'real'.\n" +
            "Responda apenas 'fake' ou 'real'." +
            "Aqui estão alguns exemplos de notícias para que você entenda o contexto. \n"
            + exemplos_str +
            f"""Agora, classifique a seguinte notícia:
            Notícia: {texto}
            Classifique: """
        )
    try:
        # pdb.set_trace()
        response = requests.post(
            url_ollama,
            json={
                "prompt": prompt,
                "model": modelo_ollama,
                "stream": False,
		"options": {"num_ctx":8192}
            },
            timeout=60
        )
        response.raise_for_status()

        # Dependendo da versão/config do Ollama, a resposta pode variar.
        # Supondo que a resposta tenha chave 'choices' -> [{'text': 'fake'}]
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            conteudo = data["choices"][0].get("text", "").strip().lower()
        else:
            conteudo = data.get("text", "").strip().lower()

        # Retornamos 'fake' se na resposta houver "fake", senão 'real'
        # Se preferir checar startswith ou um regex, fique à vontade.
        if "fake" in conteudo:
            return "fake"
        else:
            return "real"
    except Exception as e:
        print("Erro ao chamar ChatGPT:", e)
        return 'real'


def gerar_arquivo_batch_um_por_texto(textos, exemplos_rotulados=None, prop: float = None, nome: str = None):
    requests = []
    cont = 1
    # pdb.set_trace()
    for txt in textos:
        if exemplos_rotulados is None:
            # Zero-shot
            prompt = (
                "Você é um modelo de IA que classifica notícias como 'fake' ou 'real'.\n"
                "Responda apenas 'fake' ou 'real'.\n\n"
                f"Notícia: {txt}\n"
                "Classifique: "
            )
        else:
            # Few-shot
            exemplos_str = ""
            for ex in exemplos_rotulados:
                exemplos_str += f"Exemplo:\nNotícia: {
                    ex['texto']}\nRótulo: {ex['rotulo']}\n\n"

            prompt = (
                "Seu objetivo é classificar notícias como 'fake' ou 'real'.\n" +
                "Responda apenas 'fake' ou 'real'." +
                "Aqui estão alguns exemplos de notícias para que você entenda o contexto. \n"
                + exemplos_str +
                f"""Agora, classifique a seguinte notícia:
                Notícia: {txt}
                Classifique: """
            )

        # Monta a estrutura de 'messages' no formato ChatCompletion
        messages = [
            {"role": "system", "content": "Você é um classificador de fake news."},
            {"role": "user", "content": prompt}
        ]

        # Adiciona no array requests
        requests.append({
            "custom_id": f"task-{cont}-{prop}",
            # Exemplo. Altere para o modelo que deseja
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": messages,
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "response_format": {
                    "type": "json_object"
                }
            }
        })
        cont += 1
    # pdb.set_trace()
    # Salva tudo em um arquivo JSON
    with open(f"./jsonl/batch_gpt_{nome}_{prop}.jsonl", "w", encoding="utf-8") as f:
        for obj in requests:
            f.write(json.dumps(obj) + '\n')

    # print(f"Gerado arquivo '{nome_arquivo}' com {len(requests)} requisições.")


def avaliar_chatgpt(textos: List[str], y_true: np.ndarray, exemplos_rotulados: List[Dict[str, str]] = None
                    ) -> Dict[str, float]:
    """
    Avalia ChatGPT em zero-shot (exemplos_rotulados=None) ou few-shot (exemplos_rotulados != None).
    """
    y_pred = []
    cont = 0
    for txt in textos:
        pred = prever_com_chatgpt(txt, exemplos_rotulados)
        y_pred.append(1 if pred == 'fake' else 0)
        cont += 1
        print(f"iteracao {cont}/{len(textos)}")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'accuracy': acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }
