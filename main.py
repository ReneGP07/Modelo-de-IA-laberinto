
import os
os.environ.setdefault("MPLBACKEND", "TkAgg")

import argparse
from pathlib import Path
import torch

from maze_rl.levels import plan_niveles
from maze_rl.control import Control
from maze_rl.maze import generar_laberinto
from maze_rl.transitions import precompute_transitions
from maze_rl.qtransfer import resize_Q
from maze_rl.training import train_level_cuda
from maze_rl.io_utils import save_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--niveles", type=int, default=10)
    parser.add_argument("--size_min", type=int, default=21)
    parser.add_argument("--size_max", type=int, default=81)
    parser.add_argument("--seed_base", type=int, default=7)
    parser.add_argument("--episodios", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32768)
    parser.add_argument("--steps_cap", type=int, default=2048)
    parser.add_argument("--watch_every", type=int, default=200)
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--out_dir", type=str, default="checkpoints_maze")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    out_dir = Path(args.out_dir)

    niveles = plan_niveles(args.niveles, args.size_min, args.size_max, args.seed_base)

    ctrl = Control()
    Q_prev = None
    old_shape = None

    for level, size, seed in niveles:
        lab, inicio, meta = generar_laberinto(size, size, semilla=seed)
        next_s, reward, done, start_idx, goal_idx = precompute_transitions(lab, inicio, meta)

        Q_init = resize_Q(Q_prev, old_shape, lab.shape) if Q_prev is not None else None
        titulo = f"Nivel {level}/{args.niveles} — {size}x{size}"

        Q_final, curve, status = train_level_cuda(
            next_s, reward, done, start_idx, goal_idx,
            lab, inicio, meta,
            ctrl=ctrl,
            init_Q=Q_init,
            episodios=args.episodios, batch=args.batch, steps_cap=args.steps_cap,
            watch_every=args.watch_every, fps=args.fps, device=device,
            titulo_prefix=titulo
        )

        if ctrl.quit_without_save:
            print("Salida solicitada: **sin guardar**.")
            return

        # Guardar siempre que se avanza de nivel (manual o por completar)
        level_info = {
            "level": level, "shape": lab.shape, "inicio": inicio, "meta": meta,
            "semilla": seed, "episodios": args.episodios, "batch": args.batch
        }
        save_checkpoint(out_dir / f"maze_q_level_{level}.pt", Q_final, level_info, curve)
        print(f"Checkpoint guardado: nivel {level}")

        # Preparar siguiente nivel
        Q_prev = Q_final
        old_shape = lab.shape

        # Si el usuario pidió salto de nivel manual, limpiamos el flag y seguimos
        if ctrl.advance_level:
            ctrl.advance_level = False

    # Guardado final consolidado
    final_info = {
        "level": niveles[-1][0], "shape": old_shape, "inicio": inicio, "meta": meta,
        "semilla": niveles[-1][2], "episodios": args.episodios, "batch": args.batch
    }
    save_checkpoint(out_dir / "final_maze_q.pt", Q_prev, final_info, curve)
    print(f"Entrenamiento completo. Guardado final en: {out_dir/'final_maze_q.pt'}")

if __name__ == "__main__":
    main()
