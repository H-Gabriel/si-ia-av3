import numpy as np
import time
import sys

n = 100 # Quantidade de indivíduos por geração
geracao_max = 100000 # Quantidade máxima de gerações
p_recombinacao = 0.9 # Probabilidade de recombinação
p_mutacao = 0.01 # Probabilidade de mutação

inicio = time.time()
solucoes = set()

def calcular_aptidao(individuo):
    pares_atacantes = 0
    for i in range(8):
        for j in range(i + 1, 8):
            pares_atacantes += 1 if individuo[i] == individuo[j] or j - i == np.abs(individuo[i] - individuo[j]) else 0
    if 28 - pares_atacantes == 28:
        individuo_str = str(individuo)
        if individuo_str not in solucoes:
            solucoes.add(individuo_str)
            print("ÓTIMO ENCONTRADO", individuo_str)
            if (len(solucoes) == 3):
                fim = time.time()
                print("Tempo de execução:", fim - inicio)
                sys.exit()
    return 28 - pares_atacantes

def recombinar(pai1, pai2):
    p1 = np.random.randint(1,8)
    p2 = np.random.randint(1,8)
    
    while p1 == p2:
        p1 = np.random.randint(1,8)
        p2 = np.random.randint(1,8)
    if p1 > p2:
        p1, p2 = p2, p1
    
    filho1 = np.concatenate((pai1[:p1], pai2[p1:p2], pai1[p2:]))
    filho2 = np.concatenate((pai2[:p1], pai1[p1:p2], pai2[p2:]))
    
    return filho1, filho2

def mutar(filho1, filho2):
    i = np.random.randint(0, 8)
    j = np.random.randint(0, 8)

    m_i = np.random.randint(0,8) # Mutação no indice 'i' do filho 1
    m_j = np.random.randint(0,8) # Mutação no indice 'j' do filho 2

    while m_i == filho1[i]:
        m_i = np.random.randint(0,8)

    while m_j == filho2[j]:
        m_j = np.random.randint(0,8)
    
    filho1[i] = m_i
    filho2[j] = m_j

    return filho1, filho2

# Definindo população inicial
P = np.random.randint(low=0, high=8, size=(n,8))

geracao_atual = 0
while geracao_atual < geracao_max:
    # Calcula a aptidão total da geração e salva a aptidão de cada 
    aptidoes = np.array([calcular_aptidao(P[i, :]) for i in range(n)])
    aptidao_total = np.sum(aptidoes)
    geracao_atual += 1
        
    S = np.empty((0,8), dtype=int) # Conjunto de seleção
    for i in range(P.shape[0]):
        j = 0
        soma = aptidoes[j]/aptidao_total
        r = np.random.uniform()
        while soma < r:
            j += 1
            soma += aptidoes[j] /aptidao_total
        S = np.concatenate((S, P[j, :].reshape(1,8)))

    # Recombinação dos individuos selecionados
    for i in range(0, n, 2):
        if np.random.random() <= p_recombinacao: # Apto a recombinar
            filho1, filho2 = recombinar(S[i, :], S[i+1, :])
            if type(filho1) == int:
                bp=1
            if np.random.random() <= 0.01: # Chance de mutação atendida
                filho1, filho2 = mutar(filho1, filho2)
            P[i] = filho1
            P[i+1] = filho2
            continue
        P[i] = S[i, :]
        P[i+1] = S[i+1, :]

fim = time.time()
print(fim - inicio)