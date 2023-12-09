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
    k = pai1.shape[1] # Quantidade de cromossomos
    p1 = np.random.randint(1, k - 1)
    p2 = np.random.randint(1, k - 1)

    while p1 == p2:
        p1 = np.random.randint(1, k - 1)
        p2 = np.random.randint(1, k - 1)
    
    if p1 > p2:
        p1, p2 = p2, p1

    filho1 = pai1[0][p1:p2+1]
    filho2 = pai2[0][p1:p2+1]

    restante1 = np.concatenate((pai1[0][p2+1:], pai1[0][:p2+1]))
    restante2 = np.concatenate((pai2[0][p2+1:], pai2[0][:p2+1]))

    temp = [x for x in restante2 if x not in filho1]
    i = p2 + 1
    while(i < k):
        filho1 = np.concatenate((filho1, np.array([temp.pop(0)])))
        i += 1
    temp = [x for x in restante2 if x not in filho1]
    filho1 = np.concatenate((temp, filho1))

    temp = [x for x in restante1 if x not in filho2]
    i = p2 + 1
    while(i < k):
        filho2 = np.concatenate((filho2, np.array([temp.pop(0)])))
        i += 1
    temp = [x for x in restante1 if x not in filho2]
    filho2 = np.concatenate((temp, filho2))

    return filho1, filho2

def mutar(individuo):
    for i in range(individuo.shape[0] - 1):
        if np.random.random() <= 0.01:
            j = np.random.randint(i+1, individuo.shape[0])
            individuo[i], individuo[j] = individuo[j], individuo[i]
    if np.random.random() <= 0.01:
        i = np.random.randint(0, individuo.shape[0] - 1)
        individuo[individuo.shape[0] - 1], individuo[i] = individuo[i], individuo[individuo.shape[0] - 1]
    return individuo

n = 50 # Quantidade de individuos por geração
geracao_max = 10000 # Quantidade máxima de gerações
p_recombinacao = 0.9
n_pontos = 30 # Pontos por região
pontos = gerar_pontos(n_pontos) # Gera os pontos
i_origem = np.random.randint(0, pontos.shape[0]) # Escolhe um indice para o ponto de origem
p_origem = pontos[i_origem,:].reshape(1,3) # Guarda a origem
pontos = np.delete(pontos, i_origem, axis=0) # Deleta a origem da "pool" de pontos a serem acessados
melhor_inaptidao = float('inf')

melhor_caminho = None
melhoria_enontrada = False
geracoes_sem_melhoria = 0

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
        if (inaptidoes[i] < melhor_inaptidao):
            melhor_inaptidao = inaptidoes[i]
            melhor_caminho = P[i,:]
            melhoria_enontrada = True
            geracoes_sem_melhoria = 0
            print("MELHORIA ENCONTRADA. GERAÇÃO:", geracao_atual, "APTIDÃO:", melhor_inaptidao)

    if melhoria_enontrada == False:
        geracoes_sem_melhoria += 1

    if (geracoes_sem_melhoria == 1000):
        break
    
    melhoria_encontrada = False

    geracao_atual += 1

    if (geracao_atual % 100 == 0):
        print("GERAÇÃO ATUAL:", geracao_atual)

    # Preservando os individuos de elite para a próxima geração
    indices_melhores = np.argsort(inaptidoes)
    P[0] = P[indices_melhores[0]]
    P[1] = P[indices_melhores[1]]

    # Conjunto de seleção de acordo com o método do torneio
    S = np.empty((n-2, pontos.shape[0]), dtype=int) # n-2 pois existem 2 individuos de elite preservados
    for i in range(n-2):
        torneio = np.random.randint(0, n, size=2) # Índice dos individuos selecionados
        if inaptidoes[torneio[0]] < inaptidoes[torneio[1]]:
            S[i] = P[torneio[0]]
            continue
        S[i] = P[torneio[1]]

    # Recombinação dos individuos
    for i in range(0, n-2, 2): # n-2 pois existem 2 individuos de elite preservados
        if np.random.random() <= p_recombinacao: # Apto a recombinar
            filho1, filho2 = recombinar(S[i].reshape(1, pontos.shape[0]), S[i+1].reshape(1, pontos.shape[0]))
            filho1 = mutar(filho1)
            filho2 = mutar(filho2)
            P[i+2] = filho1
            P[i+3] = filho2
            continue

        P[i+2] = S[i,:]
        P[i+3] = S[i+1, :]
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pontos[:,0], pontos[:,1], pontos[:,2], c='#248DD2', marker='o')
ax.scatter(p_origem[0:,0], p_origem[0:,1], p_origem[0:,2], c='green', marker='x',linewidth=3,s=30)
plt.pause(2)

p2 = pontos[melhor_caminho[0]].reshape(1,3)
line, = ax.plot([p_origem[0,0],p2[0,0]],[p_origem[0,1],p2[0,1]],[p_origem[0,2],p2[0,2]],color='k')
plt.pause(2)

for i in range(pontos.shape[0] - 1):
    p1 = pontos[melhor_caminho[i]]
    p2 = pontos[melhor_caminho[i+1]]
    line, = ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color='k')
    plt.pause(2)

p1 = pontos[melhor_caminho[pontos.shape[0] - 1]]
line, = ax.plot([p1[0],p_origem[0,0]], [p1[1],p_origem[0,1]], [p1[2],p_origem[0,2]], color='k')
plt.pause(60)

#plt.tight_layout()
plt.show()