"""
Caso 2 - Modelos de Optimizacion Industrial (II-1122)
Resolucion exacta del TSP sobre la matriz Desde-Hasta de 13 puntos
(deposito 0 + 12 clientes), tal como se trabaja en el laboratorio
de la Clase 15.

Dos motores, mismo resultado optimo:
1) held_karp(): Programacion dinamica exacta (Bellman-Held-Karp).
   No requiere solver externo, ideal para correr en Streamlit Cloud.
2) solve_mtz_milp(): Formulacion MILP con restricciones de eliminacion
   de subciclos de Miller-Tucker-Zemlin (MTZ), resuelta con PuLP/CBC.
   Replica la logica que se pide formular en AMPL en la Parte II del caso.

Ambos métodos son EXACTOS (no heurísticos): para 13 nodos garantizan
el óptimo global, igual que lo haría AMPL con un solver de MILP.
"""

from __future__ import annotations
import itertools
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class TSPResult:
    order_labels: list          # secuencia optima de etiquetas de nodo (incluye regreso al deposito)
    order_idx: list             # secuencia optima de indices (0..n-1)
    total_distance: float
    method: str
    elapsed_seconds: float
    n_variables: int = 0
    n_constraints: int = 0
    extra: dict = field(default_factory=dict)


def model_size(n: int, formulation: str = "mtz") -> tuple[int, int]:
    """
    Calcula el numero de variables y restricciones del modelo MILP del TSP
    para n nodos (deposito + clientes), util para responder la Parte I
    (pregunta 3) y la Parte III (pregunta 8: que pasa si agrego un cliente).

    Formulacion MTZ:
      Variables:  x_ij binarias para i != j           -> n*(n-1)
                  u_i continuas para i = 1..n-1        -> (n-1)
      Restricciones:
                  salida unica (sum_j x_ij = 1)        -> n
                  entrada unica (sum_i x_ij = 1)        -> n
                  eliminacion de subciclos (MTZ)        -> (n-1)*(n-2)
    """
    if formulation != "mtz":
        raise ValueError("Formulacion no soportada")
    n_vars = n * (n - 1) + (n - 1)
    n_cons = n + n + (n - 1) * (n - 2)
    return n_vars, n_cons


def held_karp(dist: np.ndarray, labels: Sequence) -> TSPResult:
    """
    Programacion dinamica exacta de Held-Karp.
    dist: matriz NxN simetrica de distancias (indice 0 = deposito).
    labels: nombres de los nodos en el mismo orden que dist.
    Complejidad O(n^2 * 2^n): perfectamente manejable para n=13.
    """
    t0 = time.time()
    n = len(dist)
    if n != len(labels):
        raise ValueError("dist y labels deben tener el mismo tamano")

    # C[(subset, k)] = (costo minimo para visitar 'subset' terminando en k, nodo anterior)
    C = {}
    for k in range(1, n):
        C[(1 << k, k)] = (dist[0][k], 0)

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for b in subset:
                bits |= 1 << b
            for k in subset:
                prev_bits = bits & ~(1 << k)
                best = None
                for m in subset:
                    if m == k:
                        continue
                    cost = C[(prev_bits, m)][0] + dist[m][k]
                    if best is None or cost < best[0]:
                        best = (cost, m)
                C[(bits, k)] = best

    # cerrar el ciclo regresando al deposito
    bits = (1 << n) - 2  # todos los bits 1..n-1 activos (bit 0 no se usa)
    best = None
    for k in range(1, n):
        cost = C[(bits, k)][0] + dist[k][0]
        if best is None or cost < best[0]:
            best = (cost, k)

    total_cost, last = best
    # reconstruir la ruta
    order_idx = [0]
    bits_r = bits
    k = last
    path = []
    while k != 0:
        path.append(k)
        prev_bits = bits_r & ~(1 << k)
        _, m = C[(bits_r, k)]
        bits_r = prev_bits
        k = m
    path.reverse()
    order_idx = [0] + path + [0]
    order_labels = [labels[i] for i in order_idx]

    elapsed = time.time() - t0
    return TSPResult(
        order_labels=order_labels,
        order_idx=order_idx,
        total_distance=float(total_cost),
        method="Held-Karp (DP exacta)",
        elapsed_seconds=elapsed,
    )


def solve_mtz_milp(dist: np.ndarray, labels: Sequence) -> TSPResult:
    """
    Formulacion MILP con restricciones MTZ (Miller-Tucker-Zemlin), la misma
    estructura que se formula en AMPL para la Parte II del caso:

        min  sum_i sum_{j!=i} d_ij * x_ij
        s.a. sum_{j!=i} x_ij = 1            para todo i      (salida unica)
             sum_{i!=j} x_ij = 1            para todo j      (entrada unica)
             u_i - u_j + n*x_ij <= n-1      para i,j = 1..n-1, i!=j  (MTZ, anti-subciclo)
             1 <= u_i <= n-1
             x_ij in {0,1}

    Requiere PuLP (incluido en requirements.txt) con el solver CBC que
    trae integrado.
    """
    import pulp

    t0 = time.time()
    n = len(dist)
    nodes = list(range(n))

    prob = pulp.LpProblem("TSP_Caso2", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in nodes for j in nodes if i != j
    }
    u = {
        i: pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n - 1)
        for i in nodes if i != 0
    }

    # Funcion objetivo
    prob += pulp.lpSum(dist[i][j] * x[(i, j)] for i in nodes for j in nodes if i != j)

    # Salida unica / entrada unica
    for i in nodes:
        prob += pulp.lpSum(x[(i, j)] for j in nodes if j != i) == 1
        prob += pulp.lpSum(x[(j, i)] for j in nodes if j != i) == 1

    # Eliminacion de subciclos (MTZ)
    for i in nodes:
        if i == 0:
            continue
        for j in nodes:
            if j == 0 or j == i:
                continue
            prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # reconstruir ruta a partir de x_ij = 1
    nxt = {}
    for (i, j), var in x.items():
        if pulp.value(var) > 0.5:
            nxt[i] = j

    order_idx = [0]
    k = nxt[0]
    while k != 0:
        order_idx.append(k)
        k = nxt[k]
    order_idx.append(0)

    total = sum(dist[order_idx[t]][order_idx[t + 1]] for t in range(len(order_idx) - 1))
    order_labels = [labels[i] for i in order_idx]
    elapsed = time.time() - t0

    n_vars, n_cons = model_size(n, "mtz")

    return TSPResult(
        order_labels=order_labels,
        order_idx=order_idx,
        total_distance=float(total),
        method="MILP - MTZ (PuLP/CBC)",
        elapsed_seconds=elapsed,
        n_variables=n_vars,
        n_constraints=n_cons,
        extra={"status": pulp.LpStatus[prob.status]},
    )
