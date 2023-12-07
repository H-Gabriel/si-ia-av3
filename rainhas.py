import numpy as np
import time
import sys

n = 100 # Quantidade de indivíduos por geração
geracao_max = 100_000_000 # Quantidade máxima de gerações
p_recombinacao = 0.9 # Probabilidade de recombinação
p_mutacao = 0.01 # Probabilidade de mutação

inicio = time.time()
solucoes = set()

def calcular_aptidao(individuo):
    pares_atacantes = 0
    diferencas = np.abs(np.subtract.outer(individuo, individuo))
    for i in range(8):
        vec = diferencas[i, i+1:]
        for j in range(vec.shape[0]):
            if vec[j] == j + 1 or vec[j] == 0:
                pares_atacantes += 1
    '''
    if 28 - pares_atacantes == 28 and str(individuo) not in solucoes:
        fim = time.time()
        print("ÓTIMO ENCONTRADO", individuo)
        print("TEMPO:", fim - inicio)
        solucoes.add(str(individuo))
    
        if (len(solucoes) == 92):
            sys.exit()
    '''

    return 28 - pares_atacantes

def recombinar(pai1, pai2):
    p1 = np.random.randint(0,8)
    p2 = np.random.randint(0,8)
    
    while p1 == p2:
        p1 = np.random.randint(1,8)
        p2 = np.random.randint(1,8)

    if p1 > p2:
        p1, p2 = p2, p1
    
    filho1 = np.concatenate((pai1[:p1], pai2[p1:p2], pai1[p2:]))
    filho2 = np.concatenate((pai2[:p1], pai1[p1:p2], pai2[p2:]))
    
    return filho1, filho2

def mutar(filho1, filho2):
    i = np.random.randint(1, 7)
    j = np.random.randint(1, 7)
    m_i = np.random.randint(0,8) # Mutação no indice 'i' do filho 1
    m_j = np.random.randint(0,8) # Mutação no indice 'j' do filho 2

    while m_i == filho1[i]:
        m_i = np.random.randint(1,7)

    while m_j == filho2[j]:
        m_j = np.random.randint(1,7)
    
    filho1[i] = m_i
    filho2[j] = m_j

    return filho1, filho2

# Definindo população inicial
P = np.random.randint(low=0, high=8, size=(n,8))

geracao_atual = 0
while geracao_atual < geracao_max:
    # Calcula a aptidão total da geração e salva a aptidão de cada 
    aptidoes = [None] * n
    for i in range(n):
        aptidoes[i] = calcular_aptidao(P[i,:])
    aptidao_total = sum(aptidoes)
    geracao_atual += 1

    if geracao_atual % 500 == 0:
        fim = time.time()
        print(fim - inicio)
        print("Geração:", geracao_atual)
        inicio = time.time()

    # Conjunto de seleção de acordo com o método da roleta
    S = np.empty((n,8), dtype=int)
    for i in range(n):
        j = 0
        soma = aptidoes[j]/aptidao_total
        r = np.random.uniform()
        while soma < r:
            j += 1
            soma += aptidoes[j] /aptidao_total

        S[i] = P[j]

    # Recombinação dos individuos selecionados
    for i in range(0, n, 2):
        if np.random.random() <= p_recombinacao: # Apto a recombinar
            filho1, filho2 = recombinar(S[i, :], S[i+1, :])
            if np.random.random() <= 0.01: # Chance de mutação atendida
                filho1, filho2 = mutar(filho1, filho2)

            P[i] = filho1
            P[i+1] = filho2
            continue

        P[i] = S[i, :]
        P[i+1] = S[i+1, :]
fim = time.time()
print(fim - inicio)