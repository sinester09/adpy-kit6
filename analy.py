import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
from datetime import datetime, timedelta
import random
import string
import os

# Configuración de estilo
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

###########################################
# PARTE 1: GENERADOR DE DATOS DE PRUEBA
###########################################

def generar_datos_prueba(num_registros=100):
    """
    Genera un conjunto de datos de prueba para el dashboard de ventas.
    
    Args:
        num_registros (int): Número de registros a generar
        
    Returns:
        DataFrame: DataFrame con los datos generados
    """
    # Definir datos de muestra
    vendedores = np.array([
        "Erika Frisancho", 
        "Carlos Mendoza", 
        "María Gutiérrez", 
        "Juan Pérez", 
        "Ana Rodríguez", 
        "Luis Gómez", 
        "Patricia Velasco"
    ])

    etapas = np.array([
        "EN evaluacion", 
        "Cotizacion enviada", 
        "Negociacion", 
        "Cierre ganado", 
        "Cierre perdido", 
        "En seguimiento", 
        "Pendiente de reunion"
    ])

    actividades = np.array([
        "Llamada inicial", 
        "Envío de propuesta", 
        "Demostración", 
        "Reunión virtual", 
        "Seguimiento", 
        "Visita presencial", 
        "Resolución de dudas"
    ])

    divisas = np.array(["PEN", "USD", "EUR"])

    programas = np.array([
        "PE PLANDCON 2024 I ONLINE",
        "PE PLANDCON 2024 II ONLINE",
        "PE MBA EJECUTIVO 2024",
        "PE MARKETING DIGITAL 2024",
        "PE FINANZAS CORPORATIVAS 2024",
        "PE GESTION DE PROYECTOS 2024 I",
        "PE GESTION DE PROYECTOS 2024 II",
        "PE RECURSOS HUMANOS 2024 ONLINE",
        "PE BUSINESS INTELLIGENCE 2024",
        "PE SUPPLY CHAIN MANAGEMENT 2024"
    ])
    
    dominios = np.array(["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "empresa.com.pe", "corporacion.com"])
    
    # Fecha actual
    fecha_actual = datetime.now()
    
    # Generar IDs únicos
    oportunidad_id = np.array([f"OPP-{i+1:04d}" for i in range(num_registros)])
    cotizacion_id = np.array([f"COT-{i+1:04d}" for i in range(num_registros)])
    visita_id = np.array([f"VIS-{i+1:04d}" for i in range(num_registros)])
    
    # Generar nombres de clientes
    clientes = np.array([generar_nombre() for _ in range(num_registros)])
    correos = np.array([generar_correo(cliente) for cliente in clientes])
    
    # Seleccionar programas, vendedores y actividades
    programas_seleccionados = np.random.choice(programas, num_registros)
    vendedores_seleccionados = np.random.choice(vendedores, num_registros)
    actividades_seleccionadas = np.random.choice(actividades, num_registros)
    
    # Generar responsables
    responsables = np.where(np.random.rand(num_registros) > 0.3, vendedores_seleccionados, np.random.choice(vendedores, num_registros))
    
    # Generar fechas
    fechas_creacion = [fecha_aleatoria(fecha_actual - timedelta(days=60), fecha_actual) for _ in range(num_registros)]
    fechas_actividad = [fecha_aleatoria(fc, fecha_actual + timedelta(days=30)) for fc in fechas_creacion]
    
    # Generar siguientes eventos
    sig_evento_opciones = ["Reunión de seguimiento", "Presentación de propuesta", 
                           "Llamada de cierre", "Demo del producto", "Negociación final",
                           "Firma de contrato", "Reunión con decisores"]
    siguientes_eventos = np.random.choice(sig_evento_opciones, num_registros)
    
    # Asignar divisas y generar montos
    divisas_seleccionadas = np.random.choice(divisas, num_registros)
    montos = np.array([round(random.uniform(500, 15000), 2) if d == "PEN"
                       else round(random.uniform(200, 5000), 2) if d == "USD"
                       else round(random.uniform(180, 4500), 2)
                       for d in divisas_seleccionadas])
    
    # Generar costos
    costos = np.round(montos * (1 - np.random.uniform(0.1, 0.4, num_registros)), 2)
    
    # Asignar etapas con distribución realista
    probabilidades_etapas = [0.25, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05]
    etapas_seleccionadas = np.random.choice(etapas, num_registros, p=probabilidades_etapas)
    
    # Generar fechas de venta
    fechas_venta = [fecha_aleatoria(fc, fecha_actual) if etapa == "Cierre ganado" else None for fc, etapa in zip(fechas_creacion, etapas_seleccionadas)]
    
    # Crear DataFrame
    datos = {
        "oportunidad_id": oportunidad_id,
        "cotizacion_id": cotizacion_id,
        "visita_id": visita_id,
        "vendedor": vendedores_seleccionados,
        "cliente": clientes,
        "correo": correos,
        "programa": programas_seleccionados,
        "actividad": actividades_seleccionadas,
        "responsable": responsables,
        "fecha_actividad": fechas_actividad,
        "siguiente_evento": siguientes_eventos,
        "divisa": divisas_seleccionadas,
        "monto": montos,
        "costo": costos,
        "etapa_cierre": etapas_seleccionadas,
        "fecha_creacion": fechas_creacion,
        "fecha_venta": fechas_venta
    }
    
    df = pd.DataFrame(datos)
    
    # Añadir caso de ejemplo
    caso_ejemplo = {
        "oportunidad_id": "PE-OPP-0000",
        "cotizacion_id": "PE-COT-0000",
        "visita_id": "PE-VIS-0000",
        "vendedor": "Erika Frisancho",
        "cliente": "Abraham",
        "correo": "Abrahamgch94@gmail.com",
        "programa": "PE PLANDCON 2024 I ONLINE",
        "actividad": "Envío de propuesta",
        "responsable": "Erika Frisancho",
        "fecha_actividad": fecha_actual + timedelta(days=3),
        "siguiente_evento": "Llamada de seguimiento",
        "divisa": "PEN",
        "monto": 1890.00,
        "costo": 1135.00,
        "etapa_cierre": "EN evaluacion",
        "fecha_creacion": fecha_actual - timedelta(days=10),
        "fecha_venta": None
    }
    
    df = pd.concat([pd.DataFrame([caso_ejemplo]), df], ignore_index=True)
    
    for col in ['fecha_actividad', 'fecha_creacion', 'fecha_venta']:
        df[col] = pd.to_datetime(df[col])
    
    return df

def fecha_aleatoria(start, end):
    """Genera una fecha aleatoria entre dos fechas dadas."""
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def generar_correo(nombre):
    """Genera un correo electrónico plausible basado en un nombre."""
    partes = nombre.lower().split()
    if len(partes) < 2:
        partes.append("user")
    dominios = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "empresa.com.pe", "corporacion.com"]
    formatos = [
        f"{partes[0]}{partes[-1][-2:]}@{random.choice(dominios)}",
        f"{partes[0][0]}{partes[-1]}@{random.choice(dominios)}",
        f"{partes[-1]}.{partes[0][0]}@{random.choice(dominios)}",
        f"{partes[0]}{random.randint(1, 99)}@{random.choice(dominios)}",
        f"{partes[-1]}{random.randint(1, 99)}@{random.choice(dominios)}"
    ]
    return random.choice(formatos)

def generar_nombre():
    """Genera un nombre completo aleatorio."""
    nombres = ["Juan", "Ana", "Carlos", "María", "Pedro", "Laura", "José", "Claudia", 
               "Miguel", "Lucía", "Fernando", "Patricia", "Javier", "Rosa", "Alberto", 
               "Silvia", "Roberto", "Carmen", "Alejandro", "Adriana", "Eduardo", "Mónica"]
    apellidos = ["García", "Rodríguez", "López", "Martínez", "González", "Pérez", 
                "Sánchez", "Ramírez", "Torres", "Flores", "Rivera", "Gómez", 
                "Díaz", "Reyes", "Cruz", "Morales", "Ortiz", "Ramos", "Vargas", "Castillo"]
    return f"{random.choice(nombres)} {random.choice(apellidos)}"


###########################################
# PARTE 2: FUNCIONES DE ANÁLISIS DE DATOS
###########################################

def limpiar_datos(df):
    """
    Limpia y prepara los datos para el análisis.
    
    Args:
        df (DataFrame): DataFrame de pandas con los datos cargados
        
    Returns:
        DataFrame: DataFrame limpio y preparado
    """
    # Crear una copia para no modificar el original
    df_limpio = df.copy()
    
    # Convertir fechas
    for col in df_limpio.columns:
        if 'fecha' in col.lower() or 'date' in col.lower():
            df_limpio[col] = pd.to_datetime(df_limpio[col], errors='coerce')
    
    # Manejar valores nulos
    # Para columnas numéricas
    cols_numericas = df_limpio.select_dtypes(include=['number']).columns
    df_limpio[cols_numericas] = df_limpio[cols_numericas].fillna(0)
    
    # Para columnas categóricas
    cols_categoricas = df_limpio.select_dtypes(include=['object']).columns
    df_limpio[cols_categoricas] = df_limpio[cols_categoricas].fillna('No especificado')
    
    return df_limpio

def calcular_metricas_vendedor(df):
    """
    Calcula métricas de rendimiento por vendedor.
    
    Args:
        df (DataFrame): DataFrame limpio
        
    Returns:
        DataFrame: DataFrame con métricas por vendedor
    """
    # Agrupar por vendedor
    metricas_vendedor = df.groupby('vendedor').agg(
        total_ventas=('monto', 'sum'),
        promedio_venta=('monto', 'mean'),
        num_ventas=('oportunidad_id', 'count'),
        cotizaciones_enviadas=('cotizacion_id', 'nunique'),
        clientes_visitados=('cliente', 'nunique')
    ).reset_index()
    
    # Calcular tasa de conversión (ventas cerradas / cotizaciones)
    ventas_cerradas = df[df['etapa_cierre'] == 'Cierre ganado'].groupby('vendedor').size().reset_index(name='ventas_cerradas')
    
    # Fusionar con el DataFrame principal
    metricas_vendedor = pd.merge(metricas_vendedor, ventas_cerradas, on='vendedor', how='left')
    metricas_vendedor['ventas_cerradas'] = metricas_vendedor['ventas_cerradas'].fillna(0)
    
    # Calcular tasa de conversión
    metricas_vendedor['tasa_conversion'] = (metricas_vendedor['ventas_cerradas'] / 
                                           metricas_vendedor['cotizaciones_enviadas'] * 100).fillna(0)
    
    return metricas_vendedor

def analizar_etapas_cierre(df):
    """
    Analiza las oportunidades por etapa de cierre.
    
    Args:
        df (DataFrame): DataFrame limpio
        
    Returns:
        DataFrame: DataFrame con análisis por etapa
    """
    # Agrupar por etapa
    etapas = df.groupby('etapa_cierre').agg(
        num_oportunidades=('oportunidad_id', 'nunique'),
        valor_total=('monto', 'sum')
    ).reset_index()
    
    # Calcular porcentaje del total
    etapas['porcentaje_valor'] = (etapas['valor_total'] / etapas['valor_total'].sum() * 100).round(2)
    
    return etapas

def calcular_rentabilidad(df):
    """
    Calcula la rentabilidad de las oportunidades.
    
    Args:
        df (DataFrame): DataFrame limpio
        
    Returns:
        DataFrame: DataFrame con análisis de rentabilidad
    """
    # Calcular margen y porcentaje
    df_rent = df.copy()
    df_rent['margen'] = df_rent['monto'] - df_rent['costo']
    df_rent['margen_porcentaje'] = (df_rent['margen'] / df_rent['monto'] * 100).round(2)
    
    # Agrupar por vendedor y cliente
    rentabilidad = df_rent.groupby(['vendedor', 'cliente']).agg(
        monto_total=('monto', 'sum'),
        costo_total=('costo', 'sum'),
        margen_total=('margen', 'sum'),
        margen_promedio=('margen_porcentaje', 'mean')
    ).reset_index()
    
    return rentabilidad

def analizar_tendencias(df, periodo='M'):
    """
    Analiza las tendencias de ventas a lo largo del tiempo.
    
    Args:
        df (DataFrame): DataFrame limpio
        periodo (str): Periodo para agrupar ('D' para día, 'W' para semana, 'M' para mes)
        
    Returns:
        DataFrame: DataFrame con análisis de tendencias
    """
    # Trabajar solo con ventas que tienen fecha
    df_tend = df.dropna(subset=['fecha_venta']).copy()
    
    if len(df_tend) == 0:
        # Si no hay fechas de venta, usar fechas de creación como alternativa
        df_tend = df.copy()
        df_tend['fecha_para_tendencia'] = df_tend['fecha_creacion']
    else:
        df_tend['fecha_para_tendencia'] = df_tend['fecha_venta']
    
    # Crear columna de periodo
    df_tend['periodo'] = df_tend['fecha_para_tendencia'].dt.to_period(periodo)
    
    # Agrupar por periodo y vendedor
    tendencias = df_tend.groupby(['periodo', 'vendedor']).agg(
        ventas=('monto', 'sum'),
        cotizaciones=('cotizacion_id', 'nunique'),
        visitas=('visita_id', 'nunique')
    ).reset_index()
    
    # Convertir periodo a fecha para graficar
    tendencias['fecha'] = tendencias['periodo'].dt.to_timestamp()
    
    return tendencias

###########################################
# PARTE 3: DASHBOARD
###########################################

def crear_dashboard(df):
    """
    Crea un dashboard interactivo con Dash.
    
    Args:
        df (DataFrame): DataFrame limpio
        
    Returns:
        Dash app: Aplicación de Dash
    """
    # Preparar datos para el dashboard
    df_limpio = limpiar_datos(df)
    metricas_vendedor = calcular_metricas_vendedor(df_limpio)
    etapas = analizar_etapas_cierre(df_limpio)
    rentabilidad = calcular_rentabilidad(df_limpio)
    tendencias = analizar_tendencias(df_limpio)
    
    # Iniciar la aplicación Dash
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Diseño del dashboard
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Dashboard de Análisis de Ventas", className="text-center mb-4"), width=12)
        ]),
        
        # Filtros
        dbc.Row([
            dbc.Col([
                html.H4("Filtros"),
                dbc.Card([
                    dbc.CardBody([
                        html.Label("Seleccionar Vendedor:"),
                        dcc.Dropdown(
                            id='dropdown-vendedor',
                            options=[{'label': 'Todos', 'value': 'Todos'}] + 
                                    [{'label': vendedor, 'value': vendedor} 
                                     for vendedor in df_limpio['vendedor'].unique()],
                            value='Todos',
                            multi=True
                        ),
                        html.Br(),
                        html.Label("Rango de Fechas:"),
                        dcc.DatePickerRange(
                            id='date-picker',
                            start_date=df_limpio['fecha_creacion'].min(),
                            end_date=df_limpio['fecha_creacion'].max(),
                            display_format='DD/MM/YYYY'
                        )
                    ])
                ])
            ], width=12)
        ]),
        
        # Métricas clave
        dbc.Row([
            dbc.Col([
                html.H4("Métricas Clave", className="text-center"),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("Total Ventas", className="card-title text-center"),
                            html.H3(id="total-ventas", className="text-center")
                        ])
                    ]), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("Cotizaciones", className="card-title text-center"),
                            html.H3(id="total-cotizaciones", className="text-center")
                        ])
                    ]), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("Clientes Visitados", className="card-title text-center"),
                            html.H3(id="total-clientes", className="text-center")
                        ])
                    ]), width=3),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("Tasa Conversión", className="card-title text-center"),
                            html.H3(id="tasa-conversion", className="text-center")
                        ])
                    ]), width=3)
                ])
            ], width=12)
        ], className="mb-4 mt-4"),
        
        # Gráficos
        dbc.Row([
            # Rendimiento por vendedor
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rendimiento por Vendedor"),
                    dbc.CardBody([
                        dcc.Graph(id="grafico-vendedores")
                    ])
                ])
            ], width=6),
            
            # Etapas de cierre
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Oportunidades por Etapa de Cierre"),
                    dbc.CardBody([
                        dcc.Graph(id="grafico-etapas")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            # Tendencias en el tiempo
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Tendencias de Ventas en el Tiempo"),
                    dbc.CardBody([
                        dcc.Graph(id="grafico-tendencias")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            # Rentabilidad
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Análisis de Rentabilidad"),
                    dbc.CardBody([
                        dcc.Graph(id="grafico-rentabilidad")
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)
    
    # Callbacks para actualizar gráficos y métricas basado en filtros
    @app.callback(
        [Output("total-ventas", "children"),
         Output("total-cotizaciones", "children"),
         Output("total-clientes", "children"),
         Output("tasa-conversion", "children"),
         Output("grafico-vendedores", "figure"),
         Output("grafico-etapas", "figure"),
         Output("grafico-tendencias", "figure"),
         Output("grafico-rentabilidad", "figure")],
        [Input("dropdown-vendedor", "value"),
         Input("date-picker", "start_date"),
         Input("date-picker", "end_date")]
    )
    def actualizar_dashboard(vendedores, fecha_inicio, fecha_fin):
        # Filtrar datos según selección
        df_filtrado = df_limpio.copy()
        
        if vendedores and 'Todos' not in vendedores:
            df_filtrado = df_filtrado[df_filtrado['vendedor'].isin(vendedores)]
            
        if fecha_inicio and fecha_fin:
            df_filtrado = df_filtrado[(df_filtrado['fecha_creacion'] >= fecha_inicio) & 
                                     (df_filtrado['fecha_creacion'] <= fecha_fin)]
        
        # Recalcular métricas con datos filtrados
        metricas = calcular_metricas_vendedor(df_filtrado)
        etapas = analizar_etapas_cierre(df_filtrado)
        rentabilidad = calcular_rentabilidad(df_filtrado)
        tendencias = analizar_tendencias(df_filtrado)
        
        # Valores para tarjetas de métricas
        total_ventas = f"${df_filtrado['monto'].sum():,.2f}"
        total_cotizaciones = f"{df_filtrado['cotizacion_id'].nunique():,}"
        total_clientes = f"{df_filtrado['cliente'].nunique():,}"
        
        # Calcular tasa de conversión
        ventas_cerradas = df_filtrado[df_filtrado['etapa_cierre'] == 'Cierre ganado']['oportunidad_id'].nunique()
        total_cotizaciones_num = df_filtrado['cotizacion_id'].nunique()
        
        if total_cotizaciones_num > 0:
            tasa = (ventas_cerradas / total_cotizaciones_num) * 100
            tasa_conversion = f"{tasa:.1f}%"
        else:
            tasa_conversion = "0%"
        
        # Gráfico de rendimiento por vendedor
        fig_vendedores = px.bar(
            metricas,
            x='vendedor',
            y='total_ventas',
            color='tasa_conversion',
            hover_data=['num_ventas', 'cotizaciones_enviadas', 'clientes_visitados'],
            labels={
                'vendedor': 'Vendedor',
                'total_ventas': 'Total de Ventas ($)',
                'tasa_conversion': 'Tasa de Conversión (%)'
            },
            title='Rendimiento por Vendedor',
            color_continuous_scale='Viridis'
        )
        
        # Gráfico de etapas de cierre
        fig_etapas = px.pie(
            etapas, 
            values='valor_total', 
            names='etapa_cierre',
            hover_data=['num_oportunidades', 'porcentaje_valor'],
            title='Distribución de Oportunidades por Etapa'
        )
        
        # Gráfico de tendencias
        fig_tendencias = px.line(
            tendencias,
            x='fecha',
            y='ventas',
            color='vendedor',
            labels={
                'fecha': 'Periodo',
                'ventas': 'Ventas ($)',
                'vendedor': 'Vendedor'
            },
            title='Tendencia de Ventas en el Tiempo'
        )
        
        # Gráfico de rentabilidad
        fig_rentabilidad = px.scatter(
            rentabilidad,
            x='monto_total',
            y='margen_promedio',
            size='margen_total',
            color='vendedor',
            hover_data=['cliente', 'monto_total', 'costo_total', 'margen_total'],
            labels={
                'monto_total': 'Monto Total ($)',
                'margen_promedio': 'Margen Promedio (%)',
                'margen_total': 'Margen Total ($)',
                'vendedor': 'Vendedor'
            },
            title='Análisis de Rentabilidad por Cliente'
        )
        
        return (total_ventas, total_cotizaciones, total_clientes, tasa_conversion,
                fig_vendedores, fig_etapas, fig_tendencias, fig_rentabilidad)
    
    return app

###########################################
# EJECUCIÓN PRINCIPAL
###########################################

if __name__ == "__main__":
    # Generar datos de prueba o cargar desde archivo
    archivo_datos = "datos_ventas_prueba.xlsx"
    
    if os.path.exists(archivo_datos):
        print(f"Cargando datos desde archivo existente: {archivo_datos}")
        df = pd.read_excel(archivo_datos)
    else:
        print("Generando nuevos datos de prueba...")
        df = generar_datos_prueba(100)
        df.to_excel(archivo_datos, index=False)
        print(f"Datos guardados en: {archivo_datos}")
    
    print(f"Datos cargados: {len(df)} registros")
    
    # Crear y ejecutar el dashboard
    app = crear_dashboard(df)
    print("Iniciando dashboard...")
    app.run_server(debug=True, host='0.0.0.0',port=8050)
    print("Dashboard iniciado en http://localhost:8050")
