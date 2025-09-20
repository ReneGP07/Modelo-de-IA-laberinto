
import random
import numpy as np

def generar_laberinto(ancho=31, alto=31, semilla=7):
    assert ancho % 2 == 1 and alto % 2 == 1
    random.seed(semilla); np.random.seed(semilla)
    lab = np.ones((alto, ancho), dtype=np.int32)

    def vecinos_candidatos(y, x):
        dirs = [(-2,0),(2,0),(0,-2),(0,2)]
        random.shuffle(dirs)
        for dy,dx in dirs:
            ny, nx = y+dy, x+dx
            if 1 <= ny < alto-1 and 1 <= nx < ancho-1 and lab[ny, nx] == 1:
                yield ny, nx, dy, dx

    stack = [(1,1)]
    lab[1,1] = 0
    while stack:
        y,x = stack[-1]
        moved = False
        for ny, nx, dy, dx in vecinos_candidatos(y,x):
            if lab[ny, nx] == 1:
                lab[y + dy//2, x + dx//2] = 0
                lab[ny, nx] = 0
                stack.append((ny,nx)); moved = True; break
        if not moved:
            stack.pop()

    # Poner bordes sólidos
    lab[0,:] = 1; lab[-1,:] = 1; lab[:,0] = 1; lab[:,-1] = 1
    inicio = (1,1); meta = (alto-2, ancho-2)
    lab[inicio] = 0; lab[meta] = 0
    return lab, inicio, meta

def estado_idx(shape, s):
    h,w = shape
    return s[0]*w + s[1]
