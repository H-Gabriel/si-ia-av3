import numpy as np
import matplotlib.pyplot as plt

def gerar_pontos(N): 
    x_partition = np.random.uniform(-10, 10, size=(N,3))
    y_partition = np.random.uniform(0, 20, size=(N,3))
    z_partition = np.random.uniform(-20, 0, size=(N,3))
    w_partition = np.random.uniform(0, 20, size=(N,3))

    x1 = np.array([[20,-20,-20]])
    x1 = np.tile(x1,(N,1))
    x_partition = x_partition+x1

    x1 = np.array([[-20,20,20]])
    x1 = np.tile(x1,(N,1))
    y_partition = y_partition+x1

    x1 = np.array([[-20,20,-20]])
    x1 = np.tile(x1,(N,1))
    z_partition = z_partition+x1

    x1 = np.array([[20,20,-20]])
    x1 = np.tile(x1,(N,1))
    w_partition = w_partition+x1   
    return np.concatenate((x_partition,y_partition,z_partition,w_partition), axis=0)

def calcular_distancia(p1,p2):    
    return np.sqrt(np.sum((p1-p2)**2))

def calcular_inaptidao(individuo, pontos, p_origem):
    inaptidao = 0
    inaptidao += calcular_distancia(p_origem, pontos[individuo[0]])
    for i in range(individuo.shape[0] - 1):
        inaptidao += calcular_distancia(pontos[individuo[i]], pontos[individuo[i+1]])
    inaptidao += calcular_distancia(pontos[-1], p_origem)

    return inaptidao

def recombinar(pai1, pai2):
    p1 = np.random.randint(1, pai1.shape[1] - 1)
    p2 = np.random.randint(1, pai2.shape[1] - 1)

    while p1 == p2:
        p1 = np.random.randint(1, pai1.shape[1] - 1)
        p2 = np.random.randint(1, pai2.shape[1] - 1)
    
    if p1 > p2:
        p1, p2 = p2, p1

    filho1 = pai1[0][p1:p2+1]
    filho2 = pai2[0][p1:p2+1]

    r1 = np.concatenate((pai1[0][p2+1:], pai1[0][:p1]))
    r2 = np.concatenate((pai2[0][p2+1:], pai2[0][:p1]))

    for i in range(r1.shape[0]):
        temp = []
        while len(temp) != p1:
            if (r1[i] not in filho1):
                temp.append(r1[i])
        

    return filho1, filho2

def mutar(individuo):
    for i in range(individuo.shape[0]):
        if np.random.random() <= 0.01:
            j = np.random.randint(i+1, individuo.shape[0])
            individuo[i], individuo[j] = individuo[j], individuo[i]

def run():
    n = 50 # Quantidade de individuos por geração
    geracao_max = 10_000 # Quantidade máxima de gerações
    n_pontos = 31 # Pontos por região
    pontos = gerar_pontos(n_pontos) # Gera os pontos
    i_origem = np.random.randint(0, pontos.shape[0]) # Escolhe um indice para o ponto de origem
    p_origem = pontos[i_origem,:].reshape(1,3) # Guarda a origem
    pontos = np.delete(pontos, i_origem, axis=0) # Deleta a origem da "pool" de pontos a serem acessados

    # Definindo população inicial
    P = np.empty((n, pontos.shape[0]), dtype=int)
    for i in range(n):
        individuo = np.random.permutation(pontos.shape[0]).reshape(1,pontos.shape[0])
        P[i] = individuo

    geracao_atual = 0
    while geracao_atual < geracao_max:
        # Calcula a aptidão total da geração e salva a aptidão de cada individuo
        inaptidoes = [None] * n
        for i in range(n):
            inaptidoes[i] = calcular_inaptidao(P[i,:], pontos, p_origem)
        geracao_atual += 1

        # Conjunto de seleção de acordo com o método do torneio
        S = np.empty((n, pontos.shape[0]), dtype=int)
        for i in range(n):
            torneio = np.random.randint(0, n, size=2) # Índice dos individuos selecionados
            if inaptidoes[torneio[0]] < inaptidoes[torneio[1]]:
                S[i] = P[torneio[0]]
                continue
            S[i] = P[torneio[1]]
        
        # Recombinação dos individuos
        for i in range(0, n, 2):
            filho1, filho2 = recombinar(S[i, :].reshape(1, pontos.shape[0]), S[i+1, :].reshape(1, pontos.shape[0]))
            filho1 = mutar(filho1)
            filho2 = mutar(filho2)


run()
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pontos[:,0], pontos[:,1], pontos[:,2], c='#248DD2', marker='o')
ax.scatter(p_origem[0:,0], p_origem[0:,1], p_origem[0:,2], c='green', marker='x',linewidth=3,s=30)

#exemplo caminho a partir da origem.
p2 = pontos[0,:].reshape(1,3)
line, = ax.plot([p_origem[0,0],p2[0,0]],[p_origem[0,1],p2[0,1]],[p_origem[0,2],p2[0,2]],color='k')

plt.tight_layout()
plt.show()
'''