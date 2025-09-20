
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

from .visualize import pump, dibujar_laberinto_en_ax

def train_level_cuda(next_s, reward, done, start_idx, goal_idx,
                     lab, inicio, meta,
                     ctrl,
                     init_Q=None,
                     episodios=4000, batch=32768, alpha=0.25, gamma=0.99,
                     eps_ini=1.0, eps_fin=0.05, steps_cap=2048,
                     watch_every=200, fps=40, device="cuda",
                     titulo_prefix=""):
    h,w = lab.shape
    n_states = h*w; n_actions = reward.shape[1]
    Q = torch.zeros((n_states, n_actions), device=device, dtype=torch.float32)
    if init_Q is not None:
        Q = init_Q.to(device).clone()

    next_s = next_s.to(device); reward = reward.to(device); done = done.to(device)
    start_idx = start_idx.to(device); goal_idx = goal_idx.to(device)

    # Figura dos paneles
    plt.ion()
    fig, (ax_maze, ax_curve) = plt.subplots(1, 2, figsize=(10,5))
    ctrl.bind_to(fig)
    dibujar_laberinto_en_ax(ax_maze, lab)
    ax_maze.scatter([inicio[1]],[inicio[0]], marker="o", s=100)
    ax_maze.scatter([meta[1]],[meta[0]], marker="*", s=150)
    ax_curve.set_xlabel("Episodio"); ax_curve.set_ylabel("Recompensa media (lote)")
    ax_curve.set_title("Progreso del entrenamiento")
    line, = ax_curve.plot([], [])
    fig.suptitle(f"{titulo_prefix} Episodio 0  Paso 0")
    fig.tight_layout(); fig.show(); fig.canvas.flush_events()

    recompensas_medias = []
    eps = eps_ini; decay = (eps_ini - eps_fin) / max(1, episodios*0.8)
    s = torch.full((batch,), start_idx.item(), device=device, dtype=torch.long)

    status = "completed"

    for ep in trange(episodios, desc=f"{titulo_prefix}"):
        if ctrl.quit_without_save:
            status = "quit"; break
        if ctrl.advance_level:
            status = "advance"; break

        # Pausa interactiva
        while ctrl.paused and not ctrl.quit_without_save and not ctrl.advance_level:
            fig.suptitle(f"{titulo_prefix} — PAUSA (P para reanudar, N para pasar nivel, Q para salir sin guardar)")
            pump(fig, fps=fps)

        s.fill_(start_idx.item())
        fin = torch.zeros(batch, device=device, dtype=torch.bool)
        R = torch.zeros(batch, device=device, dtype=torch.float32)

        for t in range(steps_cap):
            if ctrl.quit_without_save or ctrl.advance_level:
                break
            # ε-greedy
            rand = torch.rand(batch, device=device)
            a_rand = torch.randint(0, n_actions, (batch,), device=device)
            a_greedy = torch.argmax(Q[s], dim=1)
            a = torch.where(rand < eps, a_rand, a_greedy)

            ns = next_s[s, a]; r = reward[s, a]; d = done[s, a]
            R += r * (~fin)

            max_next = torch.max(Q[ns], dim=1).values
            td_target = r + gamma * max_next * (~d)
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * td_target

            s = ns; fin |= d
            if fin.all(): break

        if ctrl.quit_without_save or ctrl.advance_level:
            status = "quit" if ctrl.quit_without_save else "advance"
            break

        # curva
        recompensas_medias.append(float(R.mean().item()))
        x = np.arange(len(recompensas_medias))
        line.set_data(x, recompensas_medias)
        ax_curve.relim(); ax_curve.autoscale_view(True, True, True)

        # rollout animado
        if watch_every is not None and (ep % watch_every == 0 or ep == episodios-1):
            dibujar_laberinto_en_ax(ax_maze, lab)
            ax_maze.scatter([inicio[1]],[inicio[0]], marker="o", s=100)
            ax_maze.scatter([meta[1]],[meta[0]], marker="*", s=150)
            s_run = torch.tensor([start_idx.item()], device=device, dtype=torch.long)
            last_si, repeat = -1, 0; pasos = 0
            for _ in range(h*w*4):
                if ctrl.quit_without_save or ctrl.advance_level or ctrl.paused:
                    break
                a_g = torch.argmax(Q[s_run] + 1e-6*torch.randn_like(Q[s_run]), dim=1)
                ns_run = next_s[s_run, a_g].squeeze(0)
                si = int(ns_run.item()); y,x_ = divmod(si, w)
                ax_maze.scatter([x_],[y], marker="s", s=30)
                fig.suptitle(f"{titulo_prefix} Episodio {ep}  Paso {pasos}"); pasos += 1
                pump(fig, fps=fps)
                if si == int(goal_idx.item()): break
                repeat = repeat+1 if si == last_si else 0
                last_si = si
                if repeat > 50: break
                s_run = ns_run.view(1)

        if eps > eps_fin: eps -= decay
        pump(fig, fps=fps)

    plt.ioff()
    return Q.detach(), np.array(recompensas_medias), status
