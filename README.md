
# Maze Q-Learning (Proyecto modular)

Refactor de tu script en un proyecto **modular**, con un `main` para ejecutarlo y archivos separados por responsabilidad.
Todo está en español y mantiene exactamente la misma lógica que tu código original.

## Estructura

```
maze_q_learning_project/
├─ main.py
├─ requirements.txt
├─ README.md
└─ maze_rl/
   ├─ __init__.py
   ├─ constants.py
   ├─ control.py
   ├─ maze.py
   ├─ transitions.py
   ├─ visualize.py
   ├─ qtransfer.py
   ├─ training.py
   ├─ io_utils.py
   └─ levels.py
```

## Cómo usar

1) (Opcional) Crea un entorno virtual.
2) Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3) Ejecuta el entrenamiento:
   ```bash
   python main.py --niveles 10 --size_min 21 --size_max 81 --seed_base 7 --episodios 4000 --batch 32768 --steps_cap 2048 --watch_every 200 --fps 40 --out_dir checkpoints_maze
   ```

   También puedes ejecutar como módulo:
   ```bash
   python -m maze_rl.cli  # (si creas un cli.py en el futuro)
   ```

### Controles en la ventana
- **P** o **ESPACIO**: Pausar/Reanudar
- **N** o **→**: Pasar al siguiente nivel (guarda checkpoint del nivel actual)
- **Q** o **ESC**: Salir **sin** guardar

Los checkpoints se guardan por nivel en `--out_dir`.

## Notas de Backend (Matplotlib)
Se fuerza el backend **TkAgg** para la interfaz interactiva. En algunos entornos remotos o headless puede requerir `sudo apt-get install python3-tk` o usar un backend no interactivo.
