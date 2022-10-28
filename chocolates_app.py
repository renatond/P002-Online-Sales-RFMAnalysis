import re
import sys
import math
import calendar
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from PIL import Image
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from streamlit_card import card
from itertools import combinations
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# =======================================================================
# Settings
# =======================================================================

sys.path.insert(1, '/lib')

# pandas config
pd.set_option('display.float_format', lambda x: '%.1f' % x)

# streamlit config
st.set_page_config(layout='wide')

# Auxiliary Variables
today = dt.datetime(2022,9,30) #evitar distância temporal exceciva da base de dados

# =======================================================================
# Load Data
# =======================================================================

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_excel(path)
    return data

# =======================================================================
# Data transformation
# =======================================================================

# adjusting datetime format =============================================
def correct_formats(data):
    cols = [x.lower() for x in data.columns]
    cols = [x.replace(r" ", '_') for x in cols]
    cols = [x.replace(r"á", 'a') for x in cols]
    data.columns = cols
    data['data'] = pd.to_datetime(data['data']).dt.strftime('%Y-%m-%d')
    data['data'] = pd.to_datetime(data['data'])
    data['id_sku'] = data['id_sku'].astype(object)

    return data

# create new attributes
def create_new_attributes(data):
    data['mes'] = data['data'].dt.month

    # Coluna nome_mes
    data['nome_mes'] = data['data'].apply( lambda x: calendar.month_name[x.month])

    # Coluna com o % de Cacau
    data['%Cacau'] = data['sku'].apply(lambda x: re.search("\d+", x)[0] if '% Cacau' in x else 'NA')

    return data

# create datasets =========================================
def create_products_dataset(data):
    produtos = data[[ 'id_sku', 'sku', 'valor_unitario_sku', '%Cacau']].drop_duplicates(subset='id_sku', keep='first').reset_index(drop=True)
    produtos.to_csv('datasets/produtos.csv', index=False)
    
    return produtos

def create_sales_dataset(data):
    vendas = data[[ 'id_compra', 'data', 'mes', 'cliente', 'valor_total_da_compra']].drop_duplicates(subset='id_compra', keep='first').reset_index(drop=True)
    vendas.to_csv('datasets/vendas.csv', index=False)
    
    return vendas

def create_bitter_dataset(data):
    chocolates_amargos = data[data['%Cacau'] != 'NA'].reset_index(drop=True)
    chocolates_amargos['%Cacau'] = chocolates_amargos['%Cacau'].astype(int)
    chocolates_amargos = chocolates_amargos[chocolates_amargos['%Cacau'] >= 70].reset_index(drop=True)
    chocolates_amargos.to_csv('datasets/chocolates_amargos.csv', index=False)
    
    return chocolates_amargos

def target_cotumers(data):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Clientes Alvo para Campanha dos Nvos Produtos</h3>", unsafe_allow_html=True)
    combined = pd.merge(bitter_dataset['id_sku'], data, on='id_sku', how='right')
    target_clients = combined[combined['%Cacau'] != 'NA']
    target_clients['%Cacau'] = target_clients['%Cacau'].astype( int )
    target_clients = target_clients[target_clients['%Cacau'] >=70]
    target_clients.drop_duplicates(subset='id_compra', keep='first', inplace=True)
    target_clients.reset_index(drop=True, inplace=True)
    target_clients = target_clients[['cliente', 'id_compra']].groupby('cliente').count().sort_values('id_compra', ascending=False).head(50)
    target_clients.columns = ['Qtd. de Compras']
    target_clients.to_csv('datasets/target_costumers.csv')
    st.dataframe(target_clients, height=300)

    # @st.cache
    # # def convert_df(df):
    # #     # IMPORTANT: Cache the conversion to prevent computation on every rerun
    csv = target_clients.to_csv().encode('utf-8')
    st.download_button( "Download data as CSV",
                        data=csv,
                        file_name='target_clients.csv',
                        mime='text/csv')

    return None

# Measures ============================================================
def create_measures(data):
    faturamento_total = data[['id_compra', 'valor_total_da_compra', 'mes']].sort_values('id_compra').drop_duplicates(subset='id_compra',
                                                                                                              keep='first')['valor_total_da_compra'].sum()
    ticket_medio = round(faturamento_total/data['id_compra'].unique().size, 2)

    return faturamento_total, ticket_medio

# =========================================================================
# # Sidebar filters
# =========================================================================
# def sidebar_filters(data):
#     st.sidebar.title('Dataset Filters')

#     # Data Overview Filters
#     date_filter = st.sidebar.expander(label='Filtro de Data')
#     with date_filter:
#             # Date Interval ===================================
#         data['data'] = data['data'].dt.date
#         min_date = data['data'].min()
#         max_date = data['data'].max()
#         f_date = st.date_input('Select Date:', (min_date, max_date), min_value=min_date, max_value=max_date )
#     return data       


# =======================================================================
# Data Overview
# =======================================================================
def data_overview(data):
    c1, c2 = st.columns((1,40))

    with c1:    # Dengo Logo
        photo = Image.open('dashboards/misc/DENGO-LOGOTIPO_terra.png')
        st.image(photo, width=200)

    with c2:    # Opening Title
        st.markdown("<h2 style='text-align: center;'>Dashboard de Vendas 2022</h2>", unsafe_allow_html=True)
    st.markdown("""---""")

    c1, c2, c3 = st.columns((1,1,1))

    with c1:
        card(title="R$ {:,}".format(faturamento_total), text="Faturamento Total", image='dashboards/misc/logo-white.png')
    with c2:
        card(title="R$ {:,}".format(ticket_medio), text="Ticket Médio", image='')
    with c3:
        produto_mais_vendido = data[['quantidade_sku', 'sku']].groupby('sku').sum().sort_values('quantidade_sku', ascending=False).reset_index()['sku'][0]
        card(title=produto_mais_vendido, text="Produto Best Seller", image='')

def cohort_period(data):
    data['cohort_period'] = np.arange(len(data)) + 1
    return data

def costumer_behavior(data):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Comportamento dos Clientes</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns((1, 1))
    with c1:    # Histograma de Frequencias

        fig = px.histogram( data,
                            x=data[['cliente', 'id_compra']].groupby('cliente').count()['id_compra'],
                            labels={'count' : 'Qtd. Compras por Cliente', 'x' : 'Qtd. Compras'},
                            log_y=True,
                            text_auto=True,
                            # color='id_c   ompra',
                            color_discrete_sequence=['indianred'],
                            title='Distribuição de Frequência de Compras' )
        fig.update_layout(bargap=0.2)
        c1.plotly_chart( fig, use_containder_width=True )

    with c2: # Análise Cohort
        cohort = data[[ 'id_compra', 'data', 'cliente']]
        cohort['data'] = cohort['data'].apply(lambda x: x.strftime('%Y-%m'))
        cohort.set_index('cliente', inplace=True)
        cohort['cohort'] = cohort.groupby(level=0)['data'].min()
        cohort.reset_index(inplace=True)
        cohort = cohort.groupby(['cohort', 'data'])
        cohort = cohort.agg({'cliente' : pd.Series.nunique})
        cohort.rename(columns={'cliente' : 'Total de clientes'}, inplace=True)
        cohort = cohort.groupby(level=0).apply(cohort_period)
        cohort.reset_index(inplace=True)
        cohort.set_index(['cohort', 'cohort_period'], inplace=True)
        cohort_group_size = cohort['Total de clientes'].groupby(level=0).first()
        cohort_data = cohort.reset_index()
        cohort_counts = cohort_data.pivot(index='cohort', 
                                   columns='cohort_period',
                                   values='Total de clientes')
        cohort_sizes = cohort_counts.iloc[:,0]
        retention = cohort_counts.divide(cohort_sizes, axis = 0)
        retention = retention.round(3)*100

        user_retention = cohort['Total de clientes'].unstack(0).divide(cohort_group_size, axis=1)

        
        fig = px.imshow(user_retention.T,
                        title='Análise Cohort',
                        labels=dict(x="Cohort Period", y="Mês de Entrada", color="Retenção"),
                        text_auto=".2%",
                        aspect="auto",
                        color_continuous_scale='RdBu_r')
        c2.plotly_chart( fig, use_containder_width=True )
    return data

def top_five_products(data): # Top 5 combinações d eprodutos vendido juntos
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Análise de Produtos</h3>", unsafe_allow_html=True)
    bundle = data[data['id_compra'].duplicated(keep=False)];
    bundle['bundle'] = bundle.groupby('id_compra')['sku'].transform(lambda x: ', '.join(x));
    bundle = bundle[['id_compra', 'bundle']].drop_duplicates()
    count = Counter()

    for row in bundle['bundle']:
        row_list = row.split(',')
        count.update(Counter(combinations(row_list,2)))

    top_five = pd.DataFrame(count.most_common(5))
    top_five.columns = ['Combinação de Produtos', 'Qtd. Vendas'] 
    top_five['Combinação de Produtos'] = top_five['Combinação de Produtos'].astype( str )
    top_five['Combinação de Produtos'] = top_five['Combinação de Produtos'].str.replace(r"['''''()]","")
    top_five.to_csv('datasets/bundle.csv')
    
    fig = px.bar(top_five.sort_values('Qtd. Vendas', ascending=True),
                 title='Top 5 Produto Vendidos Juntos',
                 x="Qtd. Vendas",
                 y="Combinação de Produtos",
                 orientation='h',
                 width=1100,
                 height=400,
                 color_discrete_sequence=['saddlebrown'],
                 text_auto=True)
    fig.update_layout(bargap=0.1)
    fig.update_traces(width=0.5)
    st.plotly_chart( fig, use_containder_width=True )

    return None

def calcular_wcss(data): #Calcula a inércia de distribuição dos pontos de um Cluster
    wcss = []
    for k in range(1,10):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X=data)
        data['Clusters']=kmeans.labels_
        wcss.append(kmeans.inertia_)
    return wcss

def numero_otimo_clusters(wcss): #Define o número ótimo de Clusters para um determinado conjunto de pontos
    x1, y1 = 0,wcss[0]
    x2, y2 = 20,wcss[len(wcss)-1]
    
    distancia = []
    for i in range(len(wcss)):
        x0 = i
        y0 = wcss[i]
        numerador = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominador = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distancia.append(numerador/denominador)
    return distancia.index(max(distancia))

def ordenador_cluster(cluster_nome,target_nome,df, ascending=True): # Ordena o CLuster de acordo com critéro ojetivo
    agrupado_por_cluster = df.groupby(cluster_nome)[target_nome].mean().reset_index()
    
    if ascending == True:
        agrupado_por_cluster_ordenado = agrupado_por_cluster.sort_values(by=target_nome,ascending=True).reset_index(drop=True)
    else:
        agrupado_por_cluster_ordenado = agrupado_por_cluster.sort_values(by=target_nome,ascending=False).reset_index(drop=True)

    agrupado_por_cluster_ordenado['index'] = agrupado_por_cluster_ordenado.index
    juntando_cluster = pd.merge(df,agrupado_por_cluster_ordenado[[cluster_nome,'index']],on=cluster_nome)
    removendo_dandos = juntando_cluster.drop([cluster_nome],axis=1)
    df_final = removendo_dandos.rename(columns={'index':cluster_nome})
    return df_final

def rfm_analisys(data):
    segmentation = data[['cliente','data','id_compra','valor_total_da_compra']]
    segmentation.drop_duplicates(subset='id_compra', keep='first', inplace=True);

    rfm= segmentation.groupby('cliente').agg({'data': lambda date: (today - date.max()).days,
                                            'id_compra': lambda num: len(num),
                                            'valor_total_da_compra': lambda price: price.sum()})
    rfm.columns=['recency','frequency','monetary']
    rfm['recency'] = rfm['recency'].astype(int)
    rfm.sort_values('recency', ascending=False)  

    # CLusterização da Recência
    df_recencia = rfm[['recency']]
    recency_squared_sum = calcular_wcss(df_recencia)
    n = numero_otimo_clusters(recency_squared_sum)
    kmeans=KMeans(n_clusters=n)
    rfm['RecenciaCluster'] = kmeans.fit_predict(df_recencia)
    rfm = ordenador_cluster('RecenciaCluster','recency',rfm, False)

    # CLusterização da Recência
    df_frequencia = rfm[['frequency']]
    frequency_squared_sum = calcular_wcss(df_frequencia)
    n = numero_otimo_clusters(frequency_squared_sum)
    kmeans = KMeans(n_clusters=n)
    rfm['FrequenciaCluster'] = kmeans.fit_predict(df_frequencia)
    rfm = ordenador_cluster('FrequenciaCluster','frequency',rfm, True)

    # CLusterização da Recência
    df_monetary = rfm[['monetary']]
    monetary_squared_sum = calcular_wcss(df_monetary)
    n = numero_otimo_clusters(monetary_squared_sum)
    kmeans = KMeans(n_clusters=n)
    rfm['monetaryCluster'] = kmeans.fit_predict(df_monetary)
    rfm = ordenador_cluster('monetaryCluster','monetary',rfm, True)

    # Score de RFM
    rfm['Score'] = rfm['RecenciaCluster'] +rfm['FrequenciaCluster']+rfm['monetaryCluster']

    # Segmentação por Valor
    rfm['Seguimento'] = rfm['Score'].apply( lambda x: 'Inativo' if x < 3 else
                                                    'Baixo Valor' if x < 5 else
                                                    'Medio Valor' if x < 7 else 'Alto Valor')
    # Export RFM Dataset
    rfm.to_csv('datasets/RFM.csv')
    return rfm

def plot_seguimento(x,y,data):
    fig = px.scatter(data,
                     x,
                     y,
                     color=data['Seguimento'],
                     size=data['Score'])
    return fig

def seguimentaton(data):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Análise RFM</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns((1,1))

    with c1:
        fig = plot_seguimento('recency','frequency',rfm)
        fig.update_layout(title='Recência x Frequência')
        c1.plotly_chart( fig, use_containder_width=True )

    with c2:
        fig = plot_seguimento('frequency', 'monetary',rfm)
        fig.update_layout(title='Frequência x Gasto')
        c2.plotly_chart( fig, use_containder_width=True )

    with c1:
        fig = plot_seguimento('recency','monetary',rfm)
        fig.update_layout(title='Recência x Gasto')
        c1.plotly_chart( fig, use_containder_width=True )
    with c2:
        rfm_percentual  = rfm[['Seguimento', 'recency']].groupby('Seguimento').count().reset_index()
        rfm_percentual['recency'] = rfm_percentual['recency'].apply(lambda x: round(x/rfm_percentual['recency'].sum(), 4))
        fig = px.bar(   rfm_percentual,
                        x='Seguimento',
                        # labels={'nome_mes' : 'Mês'},
                        y=rfm_percentual['recency'],
                        text_auto='2.2%',
                        color='Seguimento',
                        # color_continuous_scale=px.colors.sequential.YlOrRd,
                        title='Distribuição de Clientes' )
        fig.update_layout(bargap=0.2)
        c2.plotly_chart( fig, use_containder_width=True )




        # fig = px.histogram( rfm,
        #                     x=rfm['Seguimento'],
        #                     # labels={'count' : 'Qtd. Compras por Cliente', 'x' : 'Qtd. Compras'},
        #                     text_auto=True,
        #                     color=rfm['Seguimento'],
        #                     title='Distribuição de Clientes por Seguimento' )
        # fig.update_layout(bargap=0.2)
        # c2.plotly_chart( fig, use_containder_width=True )


def monthly_distribution(data):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Distribuição Mensal</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns((1,1))

    with c1:    # Monthly Revenue
        faturamento_mensal =  data[['id_compra', 'valor_total_da_compra', 'mes', 'nome_mes']].sort_values('id_compra').drop_duplicates(subset='id_compra',
                                                                                                          keep='first')[['valor_total_da_compra', 'mes', 'nome_mes']].groupby(['mes', 'nome_mes']).sum().reset_index()

        faturamento_mensal.columns = ['mes', 'nome_mes', 'Faturamento']
        mes = faturamento_mensal['nome_mes']
        faturamento = faturamento_mensal['Faturamento']
        fig = px.bar(   faturamento_mensal,
                        x=mes,
                        labels={'nome_mes' : 'Mês'},
                        y='Faturamento',
                        text_auto=True,
                        color='Faturamento',
                        color_continuous_scale=px.colors.sequential.Cividis_r,
                        title='Faturamento Mensal' )
        fig.update_layout(bargap=0.2)
        c1.plotly_chart( fig, use_containder_width=True )

    with c2:    # Houses per bathrooms
        vendas_mensal =  data[['id_compra', 'valor_total_da_compra', 'mes', 'nome_mes']].sort_values('id_compra').drop_duplicates(subset='id_compra',
                                                                                                                  keep='first')[['id_compra', 'mes', 'nome_mes']].groupby(['mes', 'nome_mes']).count().reset_index()

        vendas_mensal.columns = ['mes', 'nome_mes', 'vendas']
        mes = vendas_mensal['nome_mes']
        vendas_mensais = vendas_mensal['vendas']
        fig = px.bar(   vendas_mensal,
                        x=mes,
                        labels={'nome_mes' : 'Mês'},
                        y='vendas',
                        text_auto=True,
                        color='vendas',
                        color_continuous_scale=px.colors.sequential.Cividis_r,
                        title='Vendas Mensais' )
        fig.update_layout(bargap=0.2)
        c2.plotly_chart( fig, use_containder_width=True )
    return None

if __name__ == '__main__':
    # =======================================================================================
    # Extraction
    # =======================================================================================

    # load dataset
    data = get_data('datasets/dengo_base.xlsx')

    # =======================================================================================
    # Transformation
    # =======================================================================================

    # Correct formatS
    correct_formats(data)

    # crete new attributes
    create_new_attributes(data)

    # crete datasetS
    # create_products_dataset(data)
    products_dataset = create_products_dataset(data)

    # create_sales_dataset(data)
    vendas = create_sales_dataset(data)

    # create_bitter_dataset(products_dataset)
    bitter_dataset = create_bitter_dataset(products_dataset)

    # sidebar filters
    # sidebar_filters(data)

    # Measures
    faturamento_total, ticket_medio =  create_measures(data)

    # # =======================================================================================
    # # Load
    # # =======================================================================================

    #  Dashboarding
    data_overview(data)
    monthly_distribution(data)
    costumer_behavior(vendas)
    top_five_products(data)
    rfm = rfm_analisys(data)
    seguimentaton(data)
    target_cotumers(data)

