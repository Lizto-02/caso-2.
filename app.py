"""
Caso 2 - Camino mas corto + TSP exacto
Modelos de Optimizacion Industrial (II-1122) - UCR Sede Alajuela
Clase 15 - Prof. David Benavides

App de Streamlit para resolver el TSP exacto sobre la matriz Desde-Hasta
de 13 puntos (deposito 0 + 12 clientes) y explorar los escenarios de la
Parte III del caso.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(str(Path(__file__).parent))
from src.tsp import held_karp, solve_mtz_milp, model_size

st.set_page_config(page_title="Caso 2 - TSP exacto", page_icon="🚚", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "distance_matrix.csv"


@st.cache_data
def load_matrix():
    df = pd.read_csv(DATA_PATH, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def df_to_dist(df: pd.DataFrame):
    labels = list(df.index)
    dist = df.to_numpy(dtype=float)
    return dist, labels


def draw_route(labels, order_idx, title="Ruta optima"):
    G = nx.Graph()
    G.add_nodes_from(labels)
    pos = nx.circular_layout(labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color="#1C7293", node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", ax=ax)

    route_edges = [(labels[order_idx[i]], labels[order_idx[i + 1]]) for i in range(len(order_idx) - 1)]
    G.add_edges_from(route_edges)
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color="#F96167", width=2.5, ax=ax)

    # resaltar el deposito
    depot_label = labels[0]
    nx.draw_networkx_nodes(G, pos, nodelist=[depot_label], node_color="#21295C",
                            node_size=900, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    return fig


st.title("🚚 Caso 2 · Ruteo con TSP exacto")
st.caption(
    "Clase 15 · II-1122 Modelos de Optimizacion Industrial · UCR Sede Alajuela · "
    "Prof. David Benavides"
)

st.markdown(
    """
Una empresa de distribucion sale del **deposito (nodo 0)**, visita a sus **12 clientes**
y regresa, recorriendo la **menor distancia total**. La matriz de distancias ya esta
reducida (etapa de caminos mas cortos resuelta) y lista para formular el TSP, igual
que se plantea en el laboratorio de la Clase 15.
"""
)

df = load_matrix()
dist, labels = df_to_dist(df)
n = len(labels)

tab_matriz, tab_solver, tab_escenarios, tab_modelo = st.tabs(
    ["📋 Matriz de distancias", "🧮 Resolver TSP", "🔁 Escenarios (Parte III)", "📐 Tamano del modelo"]
)

# ---------------------------------------------------------------------------
with tab_matriz:
    st.subheader("Matriz Desde-Hasta (13 x 13)")
    st.dataframe(df, use_container_width=True)
    st.caption("Distancias en km · d(i,j) = d(j,i) · diagonal = 0 · fila/columna 0 = deposito.")

# ---------------------------------------------------------------------------
with tab_solver:
    st.subheader("Resolver el TSP de forma exacta")
    metodo = st.radio(
        "Metodo de solucion",
        ["Held-Karp (programacion dinamica)", "MILP con restricciones MTZ (PuLP/CBC)"],
        help="Ambos son metodos EXACTOS: garantizan el optimo global, igual que un solver "
             "de MILP en AMPL.",
    )

    if st.button("Resolver", type="primary"):
        with st.spinner("Resolviendo..."):
            if metodo.startswith("Held"):
                result = held_karp(dist, labels)
            else:
                try:
                    result = solve_mtz_milp(dist, labels)
                except Exception as e:
                    st.error(f"No se pudo resolver con PuLP/CBC: {e}")
                    st.stop()

        st.success(f"Distancia total optima: **{result.total_distance:.0f} km**")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Secuencia de visita**")
            st.write(" → ".join(result.order_labels))
            st.markdown(f"**Metodo:** {result.method}")
            st.markdown(f"**Tiempo de computo:** {result.elapsed_seconds:.4f} s")
            if result.n_variables:
                st.markdown(f"**Variables del modelo:** {result.n_variables}")
                st.markdown(f"**Restricciones del modelo:** {result.n_constraints}")

        with col2:
            fig = draw_route(labels, result.order_idx)
            st.pyplot(fig)

        st.session_state["last_result"] = result

# ---------------------------------------------------------------------------
with tab_escenarios:
    st.subheader("Explorar escenarios de la Parte III")

    st.markdown("#### 9. Cierre temporal de una carretera (aumento de distancia)")
    c1, c2, c3 = st.columns(3)
    with c1:
        nodo_a = st.selectbox("Nodo A", labels, key="nodo_a")
    with c2:
        nodo_b = st.selectbox("Nodo B", [l for l in labels if l != nodo_a], key="nodo_b")
    with c3:
        nuevo_valor = st.number_input("Nueva distancia (km)", min_value=0.0,
                                       value=float(df.loc[nodo_a, nodo_b]), step=1.0)

    if st.button("Aplicar cambio y resolver"):
        df_mod = df.copy()
        df_mod.loc[nodo_a, nodo_b] = nuevo_valor
        df_mod.loc[nodo_b, nodo_a] = nuevo_valor
        dist_mod, labels_mod = df_to_dist(df_mod)
        with st.spinner("Resolviendo con la distancia modificada..."):
            result_mod = held_karp(dist_mod, labels_mod)
        st.info(
            f"Con d({nodo_a},{nodo_b}) = {nuevo_valor:.0f} km, la nueva ruta optima es:"
        )
        st.write(" → ".join(result_mod.order_labels))
        st.write(f"Nueva distancia total: **{result_mod.total_distance:.0f} km**")
        fig = draw_route(labels_mod, result_mod.order_idx, "Ruta tras el cambio")
        st.pyplot(fig)

    st.divider()
    st.markdown("#### 8. Agregar un cliente nuevo")
    st.caption(
        "Ingrese el nombre del nuevo cliente y su distancia a cada punto existente "
        "para ver como crece el modelo y la ruta optima."
    )
    nuevo_id = st.text_input("Identificador del nuevo cliente", value="NEW")
    nuevas_distancias = {}
    cols = st.columns(4)
    for i, lab in enumerate(labels):
        with cols[i % 4]:
            nuevas_distancias[lab] = st.number_input(
                f"d({nuevo_id}, {lab})", min_value=0.0, value=20.0, step=1.0, key=f"add_{lab}"
            )

    if st.button("Agregar cliente y resolver"):
        df_big = df.copy()
        df_big[nuevo_id] = [nuevas_distancias[l] for l in labels]
        df_big.loc[nuevo_id] = [nuevas_distancias[l] for l in labels] + [0.0]
        dist_big, labels_big = df_to_dist(df_big)
        with st.spinner("Resolviendo el modelo ampliado..."):
            result_big = held_karp(dist_big, labels_big)
        vars_before, cons_before = model_size(n, "mtz")
        vars_after, cons_after = model_size(n + 1, "mtz")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Variables (antes → despues)", f"{vars_before} → {vars_after}")
        with c2:
            st.metric("Restricciones (antes → despues)", f"{cons_before} → {cons_after}")
        st.write(" → ".join(result_big.order_labels))
        st.write(f"Nueva distancia total: **{result_big.total_distance:.0f} km**")
        fig = draw_route(labels_big, result_big.order_idx, "Ruta con el cliente nuevo")
        st.pyplot(fig)

# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("Tamano del modelo MILP (formulacion MTZ)")
    st.markdown(
        "Util para responder la **pregunta 3** (Parte I) y la **pregunta 8** (Parte III)."
    )
    n_range = st.slider("Numero de nodos (deposito + clientes)", min_value=4, max_value=25, value=n)
    v, c = model_size(n_range, "mtz")
    col1, col2 = st.columns(2)
    col1.metric("Variables", v)
    col2.metric("Restricciones", c)

    sizes = list(range(4, 21))
    vs = [model_size(s, "mtz")[0] for s in sizes]
    cs = [model_size(s, "mtz")[1] for s in sizes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, vs, marker="o", label="Variables", color="#1C7293")
    ax.plot(sizes, cs, marker="s", label="Restricciones", color="#F96167")
    ax.set_xlabel("Nodos (n)")
    ax.set_ylabel("Cantidad")
    ax.set_title("Crecimiento del modelo MTZ segun el numero de nodos")
    ax.legend()
    st.pyplot(fig)

st.divider()
st.caption(
    "Caso 2 · II-1122 Modelos de Optimizacion Industrial · UCR Sede Alajuela · "
    "Herramienta de apoyo para el laboratorio de la Clase 15."
)
