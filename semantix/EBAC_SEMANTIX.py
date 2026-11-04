import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração visual profissional para o Seaborn/Matplotlib
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Definição do diretório raiz onde o script está sendo executado
DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Mapeamento dos nomes dos arquivos originais a serem processados
arquivos = {
    "A Pagar": "a-pagar-maua-2020.csv",
    "Despesas": "despesas-maua-2020.csv",
    "Licitacoes": "licitacoes-maua-2020.csv", 
    "Receitas": "receitas-maua-2020.csv"
}

# Dicionário para armazenar os DataFrames na memória
dataframes = {}

# Loop principal para ingestão e diagnóstico inicial dos dados
for nome_base, nome_arquivo in arquivos.items():
    print("Iniciando diagnostico da base:", nome_base)
    
    caminho_completo = os.path.join(DIRETORIO_SCRIPT, nome_arquivo)
    
    try:
        # Leitura do CSV considerando o padrão de codificação e separador governamental
        df = pd.read_csv(
            caminho_completo, 
            sep=';',              
            encoding='latin-1',   
            low_memory=False      
        )
        dataframes[nome_base] = df
        
        # Análise Exploratória de Dados (EDA) - Verificação de dimensionalidade e schema
        print("Dimensoes:", df.shape[0], "registros e", df.shape[1], "colunas.")
        print("Schema Inicial:")
        print(df.dtypes.head(5)) 
        
        # Mapeamento e quantificação de valores ausentes (Nulos)
        nulos = df.isnull().sum()
        colunas_com_nulos = nulos[nulos > 0].sort_values(ascending=False).head(5)
        
        print("Top 5 Colunas com Valores Ausentes:")
        if not colunas_com_nulos.empty:
            for col, qtd in colunas_com_nulos.items():
                perc = (qtd / len(df)) * 100
                print(col, ":", qtd, "nulos (", round(perc, 1), "%)")
        else:
            print("Nenhum valor nulo encontrado.")
            
        # Verificação de integridade referencial e duplicidades
        duplicadas = df.duplicated().sum()
        print("Linhas Duplicadas Encontradas:", duplicadas)
        print("")
        
    except FileNotFoundError:
        print("Erro: O arquivo nao foi encontrado no caminho:", caminho_completo)
        print("")

# Função para normalização do schema (conversão para snake_case e remoção de acentuação)
def padronizar_nomes_colunas(df):
    df.columns = (df.columns
                  .str.normalize('NFKD')
                  .str.encode('ascii', errors='ignore')
                  .str.decode('utf-8')
                  .str.lower()
                  .str.replace(r'[^\w\s]', '', regex=True)
                  .str.strip()
                  .str.replace(r'\s+', '_', regex=True))
    return df

# Aplicação da normalização do schema em todas as bases importadas
for nome, df in dataframes.items():
    dataframes[nome] = padronizar_nomes_colunas(df)

print("Etapa de padronizacao de colunas concluida.")
print("")

df_licitacoes = dataframes["Licitacoes"]

print("Distribuicao por Modalidade de Licitacao:")

# Cálculo da distribuição estatística de processos licitatórios
if 'mod_de_licitacao' in df_licitacoes.columns:
    contagem = df_licitacoes['mod_de_licitacao'].value_counts()
    percentual = (df_licitacoes['mod_de_licitacao'].value_counts(normalize=True) * 100).round(2)
    
    resumo_modalidade = pd.DataFrame({
        'Quantidade': contagem,
        'Porcentagem': percentual
    })
    
    print(resumo_modalidade)
else:
    print("Coluna de modalidade nao encontrada. Verifique as colunas disponiveis:")
    print(df_licitacoes.columns.tolist()) 

# ==============================================================================
# 6. ANALISE FINANCEIRA - DESPESAS POR ORGAO
# ==============================================================================
print("\n------------------------------------------------------------")
print("TOP 10 ORGAOS COM MAIOR VOLUME DE DESPESAS")
print("------------------------------------------------------------")

df_despesas = dataframes["Despesas"]

# Identificação dinâmica da coluna referente a valores monetários
colunas_valor = [col for col in df_despesas.columns if 'vl_' in col or 'valor' in col]

if colunas_valor:
    coluna_alvo = colunas_valor[0]
    
    # Higienização de strings monetárias para formato internacional
    df_despesas[coluna_alvo] = (
        df_despesas[coluna_alvo]
        .astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    
    # Casting da coluna para formato numérico (float) e tratamento de anomalias
    df_despesas[coluna_alvo] = pd.to_numeric(df_despesas[coluna_alvo], errors='coerce').fillna(0)
        
    # Agrupamento e soma das despesas consolidadas por órgão municipal
    gastos_por_orgao = df_despesas.groupby('ds_orgao')[coluna_alvo].sum().reset_index()
    
    gastos_por_orgao = gastos_por_orgao.sort_values(by=coluna_alvo, ascending=False).head(10)
    
    # Formatação de saída para o padrão monetário brasileiro (BRL) focado na visualização
    gastos_por_orgao['Gasto Total'] = gastos_por_orgao[coluna_alvo].apply(
        lambda x: f"R$ {x:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.')
    )
    
    exibicao = gastos_por_orgao[['ds_orgao', 'Gasto Total']]
    
    print(exibicao.to_string(index=False))
else:
    print("Nenhuma coluna de valor identificada automaticamente. Colunas da base de Despesas:")
    print(df_despesas.columns.tolist())


# ==============================================================================
# 6.1. EDA AVANCADA: TESTES ESTATISTICOS E ANALISE DE PERCENTIS
# ==============================================================================
print("\n------------------------------------------------------------")
print("ESTATISTICA DESCRITIVA E ANALISE DE PERCENTIS DAS DESPESAS")
print("------------------------------------------------------------")

# Tratar todas as colunas de valor financeiro da base de despesas para análise estatística
for col in colunas_valor:
    df_despesas[col] = (
        df_despesas[col]
        .astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    df_despesas[col] = pd.to_numeric(df_despesas[col], errors='coerce').fillna(0)

# Cálculo de estatísticas descritivas completas e percentis
percentis = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
estatisticas_despesas = df_despesas[colunas_valor].describe(percentiles=percentis)

print(estatisticas_despesas.round(2))

# Cálculo da assimetria (Skewness) e curtose (Kurtosis) para entender a cauda longa dos gastos
print("\nAssimetria (Skewness):")
print(df_despesas[colunas_valor].skew().round(2))

print("\nCurtose (Kurtosis):")
print(df_despesas[colunas_valor].kurt().round(2))


# ==============================================================================
# 6.2. EDA AVANCADA: GERACAO DE GRAFICOS E MAPA DE CALOR (HEATMAP)
# ==============================================================================
print("\n------------------------------------------------------------")
print("GERANDO GRAFICOS DE DIAGNOSTICO E MATRIZ DE CORRELACAO...")
print("------------------------------------------------------------")

# 1. GRAFICO DE BARRAS: TOP 10 ORGAOS POR VOLUME FINANCEIRO (COM SATURACAO DE COR)
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=gastos_por_orgao,
    x=coluna_alvo,
    y='ds_orgao',
    palette="Blues_r",  # Gradação de azul: cores mais saturadas nos valores mais altos
    hue='ds_orgao',
    legend=False
)
plt.title('Top 10 Órgãos Municipais por Volume Financeiro Executado (Mauá - 2020)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Valor Total Executado (R$)', fontsize=11)
plt.ylabel('Órgão / Secretaria', fontsize=11)

# Rótulos de dados formatados em cada barra
for container in ax.containers:
    ax.bar_label(container, fmt='R$ %.0f', padding=5, fontsize=9)

plt.tight_layout()
plt.show()


# 2. MAPA DE CALOR (HEATMAP): MATRIZ DE CORRELACAO DE VARIAVEIS NUMERICAS
# Seleciona todas as colunas numéricas (inteiros e floats) do dataset de despesas
df_numerico = df_despesas.select_dtypes(include=[np.number])

# Filtra colunas com desvio padrão maior que zero para evitar NaN na correlação
df_numerico = df_numerico.loc[:, df_numerico.std() > 0]

plt.figure(figsize=(10, 8))
matriz_correlacao = df_numerico.corr()

# Gráfico de Calor utilizando paleta divergente 'Coolwarm' com anotações de valores
sns.heatmap(
    matriz_correlacao, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    linewidths=0.5, 
    vmin=-1, 
    vmax=1,
    cbar_kws={'label': 'Coeficiente de Correlação de Pearson'}
)
plt.title('Matriz de Correlação das Variáveis Orçamentárias (Despesas Mauá)', fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# 3. GRAFICO DE DISPERSAO (SCATTER PLOT): IDENTIFICAÇÃO DE OUTLIERS MONETARIOS
if len(colunas_valor) >= 2:
    col_x = colunas_valor[0]
    col_y = colunas_valor[1]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_despesas,
        x=col_x,
        y=col_y,
        alpha=0.6,
        color="#1f77b4"
    )
    plt.xscale('log') # Escala logarítmica para visualizar discrepâncias extremas
    plt.yscale('log')
    plt.title(f'Relação Logarítmica: {col_x} vs {col_y} (Análise de Outliers)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(f'{col_x} (Escala Log)', fontsize=11)
    plt.ylabel(f'{col_y} (Escala Log)', fontsize=11)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 7. LIMPEZA FINAL PARA LOOKER STUDIO (TIPAGEM E NULOS)
# ==============================================================================

def preparar_dados_para_looker(df):
    # Conversão em lote das colunas financeiras para formato numérico compatível com ferramentas de BI
    colunas_valor = [col for col in df.columns if 'vl_' in col or 'valor' in col]
    for col in colunas_valor:
        df[col] = (df[col]
                   .astype(str)
                   .str.replace('.', '', regex=False)
                   .str.replace(',', '.', regex=False))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Substituição de valores nulos estruturais para evitar quebra de dimensões nos dashboards
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
            
    return df

# Execução da pipeline de formatação para BI em todos os DataFrames
for nome, df in dataframes.items():
    dataframes[nome] = preparar_dados_para_looker(df)

#Dados limpos

# ==============================================================================
# 8. EXPORTACAO LOCAL PARA UPLOAD MANUAL NO LOOKER STUDIO
# ==============================================================================
print("\n------------------------------------------------------------")
print("EXPORTANDO DADOS PARA O DIRETORIO LOCAL")
print("------------------------------------------------------------")

# Exportação dos dados transformados para CSVs finais no diretório da aplicação
for nome_base, df in dataframes.items():
    nome_arquivo_saida = f"{nome_base.lower().replace(' ', '_').replace('ç', 'c').replace('õ', 'o')}_limpo.csv"
    caminho_saida = os.path.join(DIRETORIO_SCRIPT, nome_arquivo_saida)
    
    df.to_csv(caminho_saida, index=False, encoding='utf-8', sep=',')
    
    print("Base", nome_base, "exportada com sucesso para:", nome_arquivo_saida)

print("\nProcesso concluido com sucesso.")