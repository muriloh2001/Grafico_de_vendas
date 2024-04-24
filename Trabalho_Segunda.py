import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Caminho para o arquivo CSV
caminho_arquivo = r'C:/Users/muril/OneDrive/Área de Trabalho/Trabalho Segunda/Vendas.csv'  # Coloque o caminho correto do seu arquivo CSV

# Carregar os dados do arquivo CSV com o separador ; e espaços ao redor dos nomes das colunas
df = pd.read_csv(caminho_arquivo, sep=';', skipinitialspace=True, encoding='ISO-8859-1')

# Limpar os espaços em branco nos nomes das colunas
df.columns = df.columns.str.strip()

# Limpar as vírgulas e converter para valores numéricos
df["Vendas (em unidades monetárias)"] = df["Vendas (em unidades monetárias)"].str.replace(',', '.').astype(float)

# Calcular a média e a mediana das vendas por ano
media_por_ano = df.groupby("Ano")["Vendas (em unidades monetárias)"].mean()
mediana_por_ano = df.groupby("Ano")["Vendas (em unidades monetárias)"].median()

# Calcular a variância e o desvio padrão das vendas de todos os meses
variancia_vendas = df["Vendas (em unidades monetárias)"].var()
desvio_padrao_vendas = df["Vendas (em unidades monetárias)"].std()

# Exibir resultados das métricas de posição
print("Média e Mediana das vendas por ano:")
for ano in media_por_ano.index:
    media = media_por_ano[ano]
    mediana = mediana_por_ano[ano]
    diferenca = mediana - media
    print(f"Ano {ano}: Média: {media:.3f}, Mediana: {mediana:.3f}, Diferença: {diferenca:.3f}")

# Exibir resultados das métricas de dispersão
print("\nVariância e Desvio Padrão das vendas de todos os meses:")
print(f"Variância: {variancia_vendas:.3f}")
print(f"Desvio Padrão: {desvio_padrao_vendas:.3f}")

# Encontrar o mês com a maior e a menor venda
maior_venda_mes = df.loc[df["Vendas (em unidades monetárias)"].idxmax()]["Mês"]
menor_venda_mes = df.loc[df["Vendas (em unidades monetárias)"].idxmin()]["Mês"]

# Calcular a média de todas as vendas
media_vendas = df["Vendas (em unidades monetárias)"].mean()

# Exibir resultados e identificar meses atípicos
print(f"Mês com a maior venda: {maior_venda_mes}")
print(f"Mês com a menor venda: {menor_venda_mes}")
print(f"Média de vendas de todos os meses: {media_vendas:.3f}")

if df["Vendas (em unidades monetárias)"].max() > media_vendas * 2 or df["Vendas (em unidades monetárias)"].min() < media_vendas / 2:
    print("Existem meses atípicos com vendas significativamente altas ou baixas em relação à média.")
else:
    print("Não há meses atípicos com vendas significativamente altas ou baixas em relação à média.")

# Calcular os quartis
quartis = df["Vendas (em unidades monetárias)"].quantile([0.25, 0.5, 0.75])

# Exibir os valores que definem cada quartil
print("\nQuartis:")
print(f"Primeiro quartil (Q1): {quartis[0.25]:.3f}")
print(f"Segundo quartil (Q2) - Mediana: {quartis[0.5]:.3f}")
print(f"Terceiro quartil (Q3): {quartis[0.75]:.3f}")

# Calcular os quartis
quartis = df["Vendas (em unidades monetárias)"].quantile([0.25, 0.5, 0.75])

# Calcular o intervalo interquartil (IQR)
IQR = quartis[0.75] - quartis[0.25]

# Exibir o intervalo interquartil
print(f"Intervalo Interquartil (IQR): {IQR:.3f}")


# Criar um array de anos e vendas
anos = np.array(df["Ano"]).reshape(-1, 1)
vendas = np.array(df["Vendas (em unidades monetárias)"])

# Inicializar o modelo de regressão linear
modelo = LinearRegression()

# Ajustar o modelo aos dados
modelo.fit(anos, vendas)

# Prever as vendas para o próximo ano (2028)
previsao_2028 = modelo.predict([[2028]])

# Exibir a previsão para o próximo ano
print(f"\nPrevisão de vendas para 2028: {previsao_2028[0]:.2f}")

# Prever as vendas para os anos existentes
previsoes = modelo.predict(anos)

# Calcular o erro médio
erro_medio = np.mean(previsoes - vendas)

# Calcular o erro percentual médio
erro_percentual_medio = np.mean((previsoes - vendas) / vendas) * 100

# Exibir os resultados
print(f"Erro médio: {erro_medio:.2f}")
print(f"Erro percentual médio: {erro_percentual_medio:.2f}%\n")

# Criar um array de anos para os próximos cinco anos
proximos_anos = np.array(range(2028, 2033)).reshape(-1, 1)

# Fazer a previsão das vendas para os próximos cinco anos
previsao_proximos_anos = modelo.predict(proximos_anos)

# Exibir a previsão para os próximos cinco anos
for i, ano in enumerate(range(2028, 2033)):
    print(f"Previsão de vendas para {ano}: {previsao_proximos_anos[i]:.2f}")
