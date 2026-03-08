import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración de la Página ---
st.set_page_config(page_title="Bank Marketing Analysis", layout="wide")

# --- Clase de Procesamiento de Datos (POO) ---
class DataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_basic_info(self):
        """Retorna información general del dataset."""
        info_df = pd.DataFrame({
            'Columna': self.df.columns,
            'Tipo': self.df.dtypes.values,
            'No Nulos': self.df.count().values,
            'Nulos': self.df.isnull().sum().values
        })
        return info_df

    def classify_variables(self):
        """Clasifica variables en numéricas y categóricas."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numeric_cols, categorical_cols

    def get_statistics(self):
        """Retorna estadísticas descriptivas."""
        return self.df.describe()

    def plot_distribution(self, column):
        """Genera un histograma con Seaborn."""
        fig, ax = plt.subplots()
        sns.histplot(self.df[column], kde=True, ax=ax, color='#2ecc71')
        ax.set_title(f'Distribución de {column}')
        return fig

    def plot_categorical(self, column):
        """Genera un gráfico de barras para variables categóricas."""
        fig, ax = plt.subplots()
        sns.countplot(data=self.df, x=column, order=self.df[column].value_counts().index, palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f'Frecuencia de {column}')
        return fig

# --- Interfaz de Streamlit ---

# Sidebar: Navegación
st.sidebar.title("📊 Navegación")
menu = st.sidebar.radio("Ir a:", ["Home", "Carga de Datos", "EDA (Análisis Exploratorio)"])

# Variable de estado para el dataset
if 'data' not in st.session_state:
    st.session_state.data = None

# --- MÓDULO 1: HOME ---
if menu == "Home":
    st.title("🏦 Bank Marketing Analysis Dashboard")
    st.sidebar.image("images.png")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Objetivo del Proyecto")
        st.write("""
        Este proyecto analiza que tan efectivas son las campañas de marketing directo de una institución bancaria. 
        El objetivo es encontrar patrones que ayuden a entender qué factores influyen en que un cliente acepte la oferta 
        (variable “y”). Con esta información se busca mejorar la tasa de éxito de las campañas, que ha bajado del 12% al 8%.
        """)
        st.subheader("Sobre el Dataset")
        st.write("""
        El conjunto de datos contiene información de **41,188 registros** con 21 variables que incluyen:
        * **Datos demográficos:** Edad, trabajo, estado civil y nivel educativo.
        * **Historial bancario:** Créditos en mora, préstamos de vivienda y préstamos personales.
        * **Datos de contacto:** Canal de comunicación (celular/fijo), mes, día y duración de la llamada.
        * **Contexto socioeconómico:** Tasa de empleo, índice de precios al consumidor y tasas de interés (Euribor).
        """)
        st.info("**Tecnologías utilizadas:** Python, Pandas, NumPy, Matplotlib, Seaborn y Streamlit.")
        
    with col2:
        st.subheader("Datos del Autor")
        st.markdown("""
        * **Nombre:** Joel Bruno Pillaca Bazan
        * **Curso:** Python for Analytics
        * **Año:** 2026
        """)

# --- MÓDULO 2: CARGA DEL DATASET ---
elif menu == "Carga de Datos":
    st.title("📂 Carga de Archivos")
    uploaded_file = st.file_uploader("Carga el archivo BankMarketing.csv", type=['csv'])
    
    if uploaded_file is not None:
        # Nota: El dataset usa ';' como separador según la vista previa
        df = pd.read_csv(uploaded_file, sep=';')
        st.session_state.data = df
        st.success("¡Archivo cargado exitosamente!")
        
        st.subheader("Vista Previa (Primeras 5 filas)")
        st.dataframe(df.head())
        
        col_a, col_b = st.columns(2)
        col_a.metric("Total de Filas", df.shape[0])
        col_b.metric("Total de Columnas", df.shape[1])
    else:
        st.warning("Por favor, sube el archivo .csv para continuar.")

# --- MÓDULO 3: EDA ---
elif menu == "EDA (Análisis Exploratorio)":
    if st.session_state.data is None:
        st.error(" Primero debes cargar los datos en el módulo 'Carga de Datos'.")
    else:
        df = st.session_state.data
        analyzer = DataAnalyzer(df)
        num_cols, cat_cols = analyzer.classify_variables()

        st.title(" Análisis Exploratorio de Datos")
        
        tabs = st.tabs(["Estructura", "Estadísticas", "Univariado", "Bivariado", "Dinámico & Hallazgos"])

        # TAB 1: Información General y Clasificación
        with tabs[0]:
            st.header("1. Información del Dataset")
            st.write(analyzer.get_basic_info())
            
            st.header("2. Clasificación de Variables")
            st.write(f"**Variables Numéricas:** {len(num_cols)}")
            st.code(num_cols)
            st.write(f"**Variables Categóricas:** {len(cat_cols)}")
            st.code(cat_cols)

        # TAB 2: Estadísticas y Nulos
        with tabs[1]:
            st.header("3. Estadísticas Descriptivas")
            st.dataframe(analyzer.get_statistics())
            st.markdown("> **Interpretación:** Observe la media y desviación estándar para detectar posibles outliers en variables como `duration` o `campaign`.")
            
            st.header("4. Análisis de Valores Faltantes")
            null_count = df.isnull().sum().sum()
            if null_count == 0:
                st.success("No se encontraron valores nulos en el dataset.")
            else:
                st.write(df.isnull().sum())

        # TAB 3: Análisis Univariado
        with tabs[2]:
            st.header("5. Distribución de Numéricas")
            selected_num = st.selectbox("Selecciona una variable numérica:", num_cols)
            st.pyplot(analyzer.plot_distribution(selected_num))
            
            st.header("6. Análisis de Categóricas")
            selected_cat = st.selectbox("Selecciona una variable categórica:", cat_cols)
            st.pyplot(analyzer.plot_categorical(selected_cat))

        # TAB 4: Análisis Bivariado
        with tabs[3]:
            st.header("7. Numérico vs Objetivo (y)")
            target_num = st.selectbox("Elegir variable para comparar con 'y':", ["age", "duration", "euribor3m"])
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='y', y=target_num, ax=ax)
            st.pyplot(fig)
            
            st.header("8. Categórico vs Objetivo (y)")
            target_cat = st.selectbox("Elegir categoría para comparar con 'y':", ["job", "education", "contact"])
            fig, ax = plt.subplots()
            pd.crosstab(df[target_cat], df['y'], normalize='index').plot(kind='bar', stacked=True, ax=ax)
            st.pyplot(fig)

        # TAB 5: Dinámico y Hallazgos
        with tabs[4]:
            st.header("9. Análisis Basado en Parámetros")
            cols_to_show = st.multiselect("Selecciona columnas para comparar:", df.columns.tolist(), default=["age", "job", "y"])
            limit = st.slider("Número de filas a mostrar:", 5, 100, 20)
            st.table(df[cols_to_show].head(limit))

            st.markdown("---") # Una línea divisoria para separar el punto 9 del 10

            # --- AQUÍ VA EL ÍTEM 10 (Dentro del tab 4 y dentro del bloque else de carga) ---
            st.header(" 10: Hallazgos Clave e Insights")

            # 1. Indicadores principales (KPIs)
            col1, col2, col3 = st.columns(3)
            total_clientes = len(df)
            # Aseguramos que 'y' sea tratado correctamente para el cálculo
            conversion_rate = (df['y'].str.lower() == 'yes').mean() * 100

            col1.metric("Total Clientes Analizados", f"{total_clientes:,}")
            col2.metric("Tasa de Conversión Actual", f"{conversion_rate:.2f}%", "-4.0%") 
            col3.metric("Promedio Duración (Seg)", f"{df['duration'].mean():.0f}s")

            st.markdown("---")

            # 2. Visualización de impacto (Análisis Crítico)
            col_graph, col_text = st.columns([2, 1])

            with col_graph:
                # Gráfico de barras: Éxito según el tipo de contacto
                fig_insight, ax_insight = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df, x='contact', hue='y', palette='magma', ax=ax_insight)
                ax_insight.set_title("Efectividad según Canal de Comunicación")
                st.pyplot(fig_insight)

            with col_text:
                st.subheader("Análisis de Canal")
                st.write("""
                Se observa que el canal Celular tiene una proporción de éxito mucho mayor que el teléfono fijo. 
                Las campañas futuras deberían priorizar la recolección de números móviles.
                """)

            # 3. Sección de Conclusiones Finales
            st.subheader("Conclusiones")

            conclusiones = [
                "**1. Calidad sobre cantidad:** Debemos enfocarnos en llamadas que superen los 3 minutos de duración.",
                "**2. Estrategia Móvil:** El contacto por celular duplica la probabilidad de éxito frente al teléfono fijo.",
                "**3. Límite de insistencia:** No llamar a un mismo cliente más de 3 veces para evitar el rechazo.",
                "**4. Público Objetivo:** Priorizar perfiles con estudios universitarios y cargos administrativos.",
                "**5. Calendario:** Evitar la saturación de llamadas en meses como mayo para no agotar a la base de datos.",
            ]

            for conc in conclusiones:
                st.info(conc)