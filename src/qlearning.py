from collections import defaultdict
import random

def q_learning(env, episodios=500, gamma=0.95, epsilon=0.1, alpha=0.1):
    Q = defaultdict(lambda: defaultdict(float))  # Q[s][a] -> valor
    acciones = env.actions()
    for _ in range(episodios):
        s = env.reset()
        terminado = False
        while not terminado:
            if random.random() < epsilon:
                a = random.choice(acciones)
            else:
                qs = Q[s]
                a = max(acciones, key=lambda x: qs[x])
            s2, r, terminado = env.step(a)
            max_q_s2 = max(Q[s2][a2] for a2 in acciones)
            Q[s][a] += alpha * (r + gamma * max_q_s2 - Q[s][a])
            s = s2
    return Q
