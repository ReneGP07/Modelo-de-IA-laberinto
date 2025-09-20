import argparse
from .envs.gridworld import GridWorld
from .qlearning import q_learning

def parse_args():
    ap = argparse.ArgumentParser(description="Q-Learning en GridWorld (demo)")
    ap.add_argument("--episodios", type=int, default=500, help="Número de episodios")
    ap.add_argument("--gamma", type=float, default=0.95, help="Factor de descuento")
    ap.add_argument("--epsilon", type=float, default=0.1, help="Exploración epsilon-greedy")
    ap.add_argument("--alpha", type=float, default=0.1, help="Tasa de aprendizaje")
    return ap.parse_args()

def main():
    args = parse_args()
    env = GridWorld(width=6, height=6, start=(0,0), goal=(5,5), obstacles={(2,2),(3,2),(3,3)})
    q = q_learning(env, episodios=args.episodios, gamma=args.gamma, epsilon=args.epsilon, alpha=args.alpha)
    print("Entrenamiento completo. Tamaño de Q:", len(q))

if __name__ == "__main__":
    main()
