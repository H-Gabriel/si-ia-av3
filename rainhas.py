import numpy as np
import time

n = 100 # Quantidade de indivíduos por geração
geracao_max = 10_000 # Quantidade máxima de gerações
p_recombinacao = 0.9 # Probabilidade de recombinação
p_mutacao = 0.01 # Probabilidade de mutação
solucoes = set()
inicio = time.time()

def calcular_aptidao(individuo):
    pares_atacantes = 0
    diferencas = np.abs(np.subtract.outer(individuo, individuo))
    for i in range(8):
        vec = diferencas[i, i+1:]
        for j in range(vec.shape[0]):
            if vec[j] == j + 1 or vec[j] == 0:
                pares_atacantes += 1

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

def mutar(individuo):
    for i in range(8):
        if np.random.random() <= 0.01:
            mutacao = np.random.randint(0,8)
            while mutacao == individuo[i]:
                mutacao = np.random.randint(0,8)
            individuo[i] = mutacao
    
    return individuo

def run():
    # Definindo população inicial
    P = np.random.randint(low=0, high=8, size=(n,8))

    geracao_atual = 0
    while geracao_atual < geracao_max:
        # Calcula a aptidão total da geração e salva a aptidão de cada individuo
        aptidoes = [None] * n
        for i in range(n):
            aptidoes[i] = calcular_aptidao(P[i,:])
            if aptidoes[i] == 28:
                if str(P[i]) not in solucoes:
                    print("ÓTIMO ENCONTRADO", P[i], "GERAÇÃO ATUAL:", geracao_atual)
                    solucoes.add(str(P[i]))
                return
        
        aptidao_total = sum(aptidoes)
        geracao_atual += 1

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
                #if np.random.random() <= 0.01: # Chance de mutação atendida
                filho1 = mutar(filho1)
                filho2 = mutar(filho2)

                P[i] = filho1
                P[i+1] = filho2
                continue

            P[i] = S[i, :]
            P[i+1] = S[i+1, :]

while (len(solucoes) < 92):
    run()

fim = time.time()
print("TEMPO GASTO AO TOTAL:", fim - inicio)