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

---

## ⚡ Aceleración con CUDA (GPU)

Este proyecto **aprovecha núcleos CUDA** cuando hay una **GPU NVIDIA** disponible y **PyTorch** lo detecta:
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# modelo.to(device)
# tensores = tensores.to(device)
```
Así, el **entrenamiento por refuerzo** (cálculo de valores Q, forward de la red y backprop) se ejecuta en GPU, 
reduciendo significativamente los tiempos de cómputo en escenarios medianos y grandes.

> Si no tienes GPU, el código continúa funcionando en CPU sin cambios.

---

## 🧠 ¿Qué modelo uso? (capas + reforzamiento)

El enfoque es **Aprendizaje por Refuerzo** (RL): el agente aprende una **política** que maximiza la recompensa esperada. 
Para aproximar la función \( Q(s,a) \), se usan **dos variantes**:

1. **Q-Table** (tabular): útil en laberintos pequeños, discretos. Guarda un valor por cada par (estado, acción).
2. **Red neuronal (DQN/MLP por capas)**: para espacios más grandes, se usa una **MLP** (perceptrón multicapa) que recibe el estado (p. ej., \((x,y)\) o el grid aplanado) y **predice los Q-values** para cada acción.
   - Ejemplo de arquitectura: `Entrada → [Linear → ReLU] × 2–3 → Linear(salida=|A|)`
   - **Pérdida**: MSE entre el objetivo de Q-Learning y la predicción de la red.
   - **Optimizador**: Adam.
   - **Exploración**: ε-greedy con **decaimiento de ε** a lo largo del entrenamiento.
   - **(Opcional)**: *Experience Replay* y *Target Network* para mayor estabilidad.

> En GPU (CUDA), el **forward/backward** de la red y el cálculo del loss se aceleran automáticamente.

---

## 🧗 Entrenamiento por niveles (10 niveles) y cómo “evoluciona” el modelo

Se aplica un **curriculum learning**: entrenamos el agente **progresivamente** en **10 niveles** de dificultad creciente 
(p. ej., laberintos más grandes, más obstáculos, recompensas más escasas). El **mismo modelo** se **continúa** entre niveles
(*fine-tuning*), por lo que **no reinicia desde cero** cada vez.

- **Niveles 1–3**: el agente aprende nociones básicas de navegación y evita paredes.
- **Niveles 4–7**: mejora la **planificación** local y reduce pasos redundantes (mejor explotación).
- **Niveles 8–10**: consolida **patrones generales** (atajos, esquinas, túneles) y **generaliza** a variaciones no vistas.

Al finalizar el **nivel 10**, el modelo ha **evolucionado**: necesita **menos exploración** para alcanzar metas nuevas,
converge más rápido y muestra **transferencia de conocimiento** entre laberintos.

---

## 🔁 ¿Para qué me sirve el entrenamiento en futuras situaciones?

- **Arranque en caliente (warm‑start)**: reutilizas los **pesos** o la **Q-Table** como punto de partida para **nuevos laberintos**.
- **Menos episodios para converger**: el agente ya “sabe” moverse, explorar y aprovechar recompensas: **aprende más rápido**.
- **Transferencia**: patrones aprendidos (evitar cuellos de botella, rodear obstáculos) **se transfieren** a entornos distintos.
- **Robustez**: ante cambios moderados (tamaño del grid, obstáculos), el desempeño **se degrada poco** y se recupera con finos ajustes.
- **Producción/Docencia**: sirve como **política base** para demos, prácticas y retos, reduciendo el tiempo de preparación.

> Consejos: guarda checkpoints por nivel (p. ej., `checkpoints/nivel_10.pt`) y exporta las mejores políticas para reuso.
