import pandas as pd
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
file_path = './dados/pre-processed.csv'
data = pd.read_csv(file_path)

# Adicionando colunas com análises de strings
data['character_count'] = data['preprocessed_news'].apply(
    len)  # Número de caracteres na manchete
data['word_count'] = data['preprocessed_news'].apply(
    lambda x: len(str(x).split()))  # Número de palavras
data['unique_word_count'] = data['preprocessed_news'].apply(
    lambda x: len(set(str(x).split())))  # Palavras únicas

# Resumo estatístico das novas colunas
summary_stats = data[['character_count', 'word_count', 'unique_word_count']].agg([
    'count', 'mean', 'std', 'min', 'max'
])

# Gráfico de distribuição do número de palavras
plt.figure(figsize=(10, 6))
plt.hist(data['word_count'], bins=50, alpha=0.7, color='green')
# plt.title('Distribuição do Número de Palavras por Manchete')
plt.xlabel('Número de Palavras')
plt.ylabel('Frequência')
plt.show()

# Comparação de caracteres e palavras por rótulo
average_stats_by_label = data.groupby(
    'label')[['word_count', 'unique_word_count']].mean()

# Gráfico de médias por rótulo
average_stats_by_label.plot(kind='bar', figsize=(10, 6), alpha=0.8)
# plt.title('Médias de Caracteres, Palavras e Palavras Únicas por Rótulo')
plt.xlabel('Rótulo')
plt.ylabel('Média')
plt.xticks(rotation=0)
plt.legend(['Número de Palavras', 'Palavras Únicas'], loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Contagem dos rótulos
label_counts = data['label'].value_counts()

# Gráfico de pizza para proporção dos rótulos
plt.figure(figsize=(8, 8))
label_counts.plot.pie(autopct='%1.1f%%', startangle=90,
                      colors=['skyblue', 'lightcoral'])
# plt.title('Proporção dos Rótulos')
plt.ylabel('')  # Remove o rótulo do eixo y
plt.show()
# Exportar o resumo estatístico
summary_stats.to_csv('summary_statistics.csv', index=False)

# Exibir as tabelas relevantes
print("Resumo Estatístico das Manchetes:")
print(summary_stats)
print("\nMédias por Rótulo:")
print(average_stats_by_label)
