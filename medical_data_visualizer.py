import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importación de los datos del examen médico
df = pd.read_csv("medical_examination.csv")

# 2. Adición de la columna 'overweight' basada en el IMC
# El peso está en kg y la altura en cm (se divide por 100 para pasarla a metros)
bmi = df["weight"] / ((df["height"] / 100) ** 2)
df["overweight"] = (bmi > 25).astype(int)

# 3. Normalización de datos (0 = bueno, 1 = malo)
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


# 4. Función para dibujar el gráfico categórico
def draw_cat_plot():
    # 5. Creación del DataFrame categórico usando pd.melt
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight",
        ],
    )

    # 6. Agrupación y reformateo para obtener los conteos (renombrando la columna de agregación)
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7. Creación del gráfico categórico en paneles con sns.catplot
    graph = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar",
    )

    # 8. Obtención de la figura para la salida estándar de la plantilla
    fig = graph.fig

    # 9. No modificar las siguientes dos líneas obligatorias
    fig.savefig("catplot.png")
    return fig


# 10. Función para dibujar el mapa de calor de correlación
def draw_heat_map():
    # 11. Limpieza de datos incorrectos mediante máscaras de filtrado por percentiles
    height_low = df["height"] >= df["height"].quantile(0.025)
    height_high = df["height"] <= df["height"].quantile(0.975)
    weight_low = df["weight"] >= df["weight"].quantile(0.025)
    weight_high = df["weight"] <= df["weight"].quantile(0.975)
    pressure_ok = df["ap_lo"] <= df["ap_hi"]

    df_heat = df[
        pressure_ok & height_low & height_high & weight_low & weight_high
    ]

    # 12. Cálculo de la matriz de correlación de Pearson
    corr = df_heat.corr(method="pearson")

    # 13. Generación de una máscara booleana para ocultar el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configuración de la figura de Matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Graficado de la matriz usando sns.heatmap con formato de un decimal
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )

    # 16. No modificar las siguientes dos líneas obligatorias
    fig.savefig("heatmap.png")
    return fig
