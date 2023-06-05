from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def proxima_permutacao(vetor):
    n = len(vetor)
    i = n - 2
    while i >= 0 and vetor[i] >= vetor[i + 1]:
        i -= 1
    if i == -1:
        return sorted(vetor)
    j = n - 1
    while j > i and vetor[j] <= vetor[i]:
        j -= 1
    vetor[i], vetor[j] = vetor[j], vetor[i]
    inicio, fim = i + 1, n - 1
    while inicio < fim:
        vetor[inicio], vetor[fim] = vetor[fim], vetor[inicio]
        inicio += 1
        fim -= 1
    return vetor


def criar_matriz_cidades(n):
    matriz = np.random.rand(n, 3)
    return matriz


def calcular_distancia(cidade1, cidade2):
    return np.linalg.norm(cidade1 - cidade2)


def criar_matriz_distancias(matriz_cidades):
    n = matriz_cidades.shape[0]
    matriz_distancias = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matriz_distancias[i, j] = calcular_distancia(matriz_cidades[i], matriz_cidades[j])

    return matriz_distancias


def calcular_custo_rota(rota, matriz_distancias):
    custo = 0
    n = len(rota)
    for i in range(n - 1):
        cidade_atual = rota[i]
        proxima_cidade = rota[i + 1]
        custo += matriz_distancias[cidade_atual, proxima_cidade]
    custo += matriz_distancias[rota[-1], rota[0]]
    return custo


def caixeiro_viajante_clusters(matriz_cidades, qtd_cidades_por_cluster):
    n = matriz_cidades.shape[0]
    qtd_clusters = int(n / qtd_cidades_por_cluster)

    modelo = KMeans(n_clusters=qtd_clusters)
    modelo.fit(matriz_cidades)
    labels = modelo.labels_
    centroides = modelo.cluster_centers_

    custo_total = 0
    melhor_rota_total = []

    for cluster_id in range(qtd_clusters):
        cidades_cluster = matriz_cidades[labels == cluster_id]
        n_cluster = len(cidades_cluster)
        if n_cluster > 1:
            matriz_distancias_cluster = criar_matriz_distancias(cidades_cluster)

            vetor_cluster = np.arange(n_cluster)
            melhor_rota_cluster = None
            menor_custo_cluster = float('inf')

            for i in range(math.factorial(n_cluster)):
                custo_cluster = calcular_custo_rota(vetor_cluster, matriz_distancias_cluster)
                if custo_cluster < menor_custo_cluster:
                    menor_custo_cluster = custo_cluster
                    melhor_rota_cluster = vetor_cluster.copy()
                vetor_cluster = proxima_permutacao(vetor_cluster)

            custo_total += menor_custo_cluster
            melhor_rota_cluster += np.min(np.where(labels == cluster_id))
            melhor_rota_total.extend(melhor_rota_cluster.tolist())

    matriz_distancias_total = criar_matriz_distancias(matriz_cidades)
    custo_total += calcular_custo_rota(melhor_rota_total, matriz_distancias_total)

    return melhor_rota_total, custo_total, centroides


# Entrada de dados
n = int(input("Digite a quantidade de cidades: "))
qtd_cidades_por_cluster = int(input("Digite a quantidade de cidades por cluster: "))

# Verificação do número mínimo de cidades para clusters
if n % qtd_cidades_por_cluster != 0:
    print("Erro: A quantidade total de cidades deve ser divisível pela quantidade de cidades por cluster.")
    exit()

# Criação das cidades aleatórias
matriz_cidades = criar_matriz_cidades(n)

# Resolução do problema do caixeiro viajante por clusters
inicio = datetime.now()
melhor_rota, menor_custo, centroides = caixeiro_viajante_clusters(matriz_cidades, qtd_cidades_por_cluster)
fim = datetime.now()

print("Melhor rota:", melhor_rota)
print("Menor custo:", menor_custo)
print("\n\ninicio:", inicio, "fim:", fim)

# Plotagem dos clusters e ligações dos centroides
plt.scatter(matriz_cidades[:, 0], matriz_cidades[:, 1], c='blue')
plt.scatter(centroides[:, 0], centroides[:, 1], marker='*', s=200, c='red')

for i in range(len(centroides)):
    cidades_cluster_i = matriz_cidades[melhor_rota == i]
    for cidade in cidades_cluster_i:
        plt.plot([cidade[0], centroides[i, 0]], [cidade[1], centroides[i, 1]], c='blue', alpha=0.2)

plt.show()
