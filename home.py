import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from joblib import load

from notebooks.src.config import DADOS_TRATADOS, MODELO_FINAL


# (Opcional) Ajusta o layout para ocupar mais espaço na tela e ficar parecido com o esboço
st.set_page_config(page_title="Previsão de Desligamento", layout="wide")

# 1. Carregamento de Dados e Modelo (Usando o CSV para evitar o erro do Parquet)
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_TRATADOS, engine='fastparquet')

@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

df = carregar_dados_limpos()
modelo = carregar_modelo()

st.title("Probabilidade de Desligamento de Funcionário")

st.markdown("---")

# ==========================================
# SEÇÃO 1: INFORMAÇÕES PESSOAIS
# ==========================================
st.subheader("Informações pessoais")
col1, col2, col3 = st.columns(3)

with col1:
    genero = st.selectbox("Gênero?", ["Masculino", "Feminino"])

with col2:
    estado_civil = st.selectbox("Estado Civil?", ["Solteiro(a)", "Casado(a)", "Divorciado(a)"])

with col3:
    area_formacao = st.selectbox("Área de Formação?", [
        "Ciências Biológicas", "Medicina", "Marketing", 
        "Curso Técnico", "Recursos Humanos", "Outros"
    ])



st.markdown("---")

# ==========================================
# SEÇÃO 2: Informacões Profissionais
# ==========================================
st.subheader("Informacões Profissionais")

# Linha superior do perfil
col4, col5, col6= st.columns(3)

with col4:
    departamento = st.selectbox("Departamento?", sorted(df['Departamento'].unique()))
    cargos_filtrados = df[df['Departamento'] == departamento]['Cargo'].unique()
    cargo = st.selectbox("Cargo?", sorted(cargos_filtrados))

    hora_extra = st.radio("Hora extra?", sorted(df['HoraExtra'].unique()))
    # Ajustado para refletir o dicionário que o pipeline espera
    viagem_negocios = st.selectbox("Viagem a negócios?", df['ViagemDeNegocios'].cat.categories)

with col5:

    anos_no_cargo = st.slider("Anos no cargo atual", 
                              min_value=int(df['AnosNoCargoAtual'].min()), 
                              max_value=int(df['AnosNoCargoAtual'].max()), 
                              value=int(df['AnosNoCargoAtual'].median()))

    anos_com_gerente = st.slider("Anos com o gerente atual", 
                                 min_value=int(df['AnosComGerenteAtual'].min()), 
                                 max_value=int(df['AnosComGerenteAtual'].max()), 
                                 value=int(df['AnosComGerenteAtual'].median()))

    anos_ultima_promocao = st.slider("Anos desde última promoção?", 
                                     min_value=int(df['AnosDesdeAUltimaPromocao'].min()), 
                                     max_value=int(df['AnosDesdeAUltimaPromocao'].max()), 
                                     value=int(df['AnosDesdeAUltimaPromocao'].median()))

    renda_mensal = st.slider("Renda Mensal?", 
                             min_value=int(df['RendaMensal'].min()), 
                             max_value=int(df['RendaMensal'].max()), 
                             value=int(df['RendaMensal'].median()))

with col6:
    # Usamos o min e max reais da base, e setamos o valor inicial na mediana
    distancia_casa = st.slider("Distância de Casa", 
                               min_value=int(df['DistanciaDeCasa'].min()), 
                               max_value=int(df['DistanciaDeCasa'].max()), 
                               value=int(df['DistanciaDeCasa'].median()))
    
    qtd_empresas = st.slider("Qtd empresas trabalhadas?", 
                             min_value=int(df['NumeroDeEmpresasQueTrabalhou'].min()), 
                             max_value=int(df['NumeroDeEmpresasQueTrabalhou'].max()), 
                             value=int(df['NumeroDeEmpresasQueTrabalhou'].median()))
    
    total_anos_trabalhados = st.slider("Total de anos trabalhados", 
                                       min_value=int(df['TotalDeAnosTrabalhados'].min()), 
                                       max_value=int(df['TotalDeAnosTrabalhados'].max()), 
                                       value=int(df['TotalDeAnosTrabalhados'].median()))    

st.markdown("---")

# ==========================================
# SEÇÃO 3: SATISFAÇÃO DO FUNCIONÁRIO
# ==========================================
st.subheader("Satisfação do Funcionário")
col11, col12, col13 = st.columns(3)

with col11:
    envolvimento_trabalho = st.slider("Envolvimento no trabalho?", 1, 4, 2)

with col12:
    satisfacao_ambiente = st.slider("Satisfação com o Ambiente", 1, 4, 2)

with col13:
    satisfacao_trabalho = st.slider("Satisfação com o trabalho", 1, 4, 2)

st.markdown("---")

# ==========================================
# LÓGICA DE PREVISÃO E PREENCHIMENTO DE NULOS
# ==========================================
if st.button("Prever Probabilidade", type="primary"):
    
    # 1. Cria um dicionário base pegando a mediana/moda, IGNORANDO a coluna alvo
    dados_previsao = {}
    for col in df.columns:
        if col != 'Desligamento': 
            # Verifica com precisão se a coluna é matemática
            if is_numeric_dtype(df[col]):
                dados_previsao[col] = df[col].median()
            else:
                dados_previsao[col] = df[col].mode()[0]
            
    # 2. Sobrescreve as variáveis do dicionário base com os valores da tela
    dados_previsao.update({
        'Genero': genero,
        'EstadoCivil': estado_civil,
        'AreaDeFormacao': area_formacao,
        'DistanciaDeCasa': distancia_casa,
        'NumeroDeEmpresasQueTrabalhou': qtd_empresas,
        'TotalDeAnosTrabalhados': total_anos_trabalhados,
        'Cargo': cargo,
        'HoraExtra': hora_extra,
        'ViagemDeNegocios': viagem_negocios,
        'Departamento': departamento,
        'AnosComGerenteAtual': anos_com_gerente,
        'AnosDesdeAUltimaPromocao': anos_ultima_promocao,
        'RendaMensal': renda_mensal,
        'AnosNoCargoAtual': anos_no_cargo,
        'EnvolvimentoNoTrabalho': envolvimento_trabalho,
        'SatisfacaoComAmbiente': satisfacao_ambiente,
        'SatisfacaoComOTrabalho': satisfacao_trabalho
    })
    
    # 3. Transforma em um DataFrame de 1 linha para passar pro modelo
    df_usuario = pd.DataFrame([dados_previsao])
    
    # 4. Faz a previsão
    try:
        probabilidade = modelo.predict_proba(df_usuario)[0][1] 
        
        st.success(f"### Risco de Desligamento: {probabilidade * 100:.1f}%")
        st.progress(probabilidade)
        
    except Exception as e:
        st.error(f"Erro ao gerar previsão: {e}")