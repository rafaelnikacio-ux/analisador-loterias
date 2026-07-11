codigo_app = '''
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="Analisador de Loterias",
    page_icon="🎲",
    layout="wide"
)

st.title("Analisador de Loterias")
st.markdown("Lotofácil | Quina | Mega-Sena")

# =========================
# FUNÇÕES DE DOWNLOAD
# =========================

@st.cache_data(ttl=3600)
def baixar_dados(loteria):
    nomes = {
        "Lotofacil": "lotofacil",
        "Quina": "quina",
        "Mega-Sena": "mega-sena"
    }
    n_dezenas = {
        "Lotofacil": 15,
        "Quina": 5,
        "Mega-Sena": 6
    }
    n_nums = {
        "Lotofacil": 25,
        "Quina": 80,
        "Mega-Sena": 60
    }

    nome_api = nomes[loteria]
    ndez = n_dezenas[loteria]
    nmax = n_nums[loteria]

    urls = [
        f"https://loteriascaixa-api.herokuapp.com/api/{nome_api}",
        f"https://raw.githubusercontent.com/maickon/free-apiloterias/master/database/{nome_api}/_todos.json",
        f"https://raw.githubusercontent.com/maickon/free-apiloterias/refs/heads/master/database/{nome_api}/_todos.json",
    ]

    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            dados = r.json()

            if isinstance(dados, dict):
                for chave in ["dados", "historico", "resultados", "items", "data"]:
                    if chave in dados and isinstance(dados[chave], list):
                        dados = dados[chave]
                        break

            registros = []
            for item in dados:
                if not isinstance(item, dict):
                    continue
                dezenas = (
                    item.get("dezenas")
                    or item.get("dezenasSorteadasOrdemSorteio")
                    or item.get("resultado")
                    or item.get("numeros")
                    or []
                )
                dezenas = sorted([int(str(d).strip()) for d in dezenas])
                concurso = (
                    item.get("concurso")
                    or item.get("numeroConcurso")
                    or item.get("numero")
                )
                data_sorteio = (
                    item.get("data")
                    or item.get("dataApuracao")
                    or item.get("dataSorteio")
                )
                if not dezenas or concurso is None:
                    continue
                registro = {
                    "concurso": int(concurso),
                    "data": data_sorteio,
                    **{f"D{i+1:02d}": dezenas[i] for i in range(len(dezenas[:ndez]))}
                }
                registros.append(registro)

            df = pd.DataFrame(registros)
            df = df.dropna(subset=["concurso"]).copy()
            df["concurso"] = df["concurso"].astype(int)
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            colunas_dez = [f"D{i:02d}" for i in range(1, ndez + 1)]
            for col in colunas_dez:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[["concurso", "data"] + colunas_dez]
            df = df.sort_values("concurso").reset_index(drop=True)

            if len(df) > 0:
                return df, ndez, nmax

        except Exception:
            continue

    return None, ndez, nmax

# =========================
# FUNÇÕES DE ANÁLISE
# =========================

def calcular_freq(df, colunas_dez, nmax):
    nums = df[colunas_dez].values.flatten()
    nums = nums[~pd.isna(nums)].astype(int)
    contagem = Counter(nums)
    freq = pd.Series({i: contagem.get(i, 0) for i in range(1, nmax + 1)})
    freq.index.name = "dezena"
    freq.name = "frequencia"
    return freq

def calcular_atrasos(df, colunas_dez, nmax):
    atrasos = {}
    ultimo = df["concurso"].max()
    for num in range(1, nmax + 1):
        mask = df[colunas_dez].apply(lambda row: num in row.values, axis=1)
        if mask.any():
            ult = df.loc[mask, "concurso"].max()
            atrasos[num] = int(ultimo - ult)
        else:
            atrasos[num] = np.nan
    return pd.Series(atrasos, name="atraso")

def calcular_pares(df, colunas_dez, ndez):
    resultado = []
    for _, row in df.iterrows():
        nums = row[colunas_dez].dropna().astype(int).tolist()
        pares = sum(1 for n in nums if n % 2 == 0)
        resultado.append(pares)
    return pd.Series(resultado, name="pares")

def calcular_soma(df, colunas_dez):
    return df[colunas_dez].sum(axis=1)

# =========================
# GERADOR DE JOGOS
# =========================

def gerar_jogos(freq, ndez, nmax, n_jogos=10, peso_freq=0.70,
                faixa_pares=None, faixa_soma=None, max_seq=2):
    numeros = list(range(1, nmax + 1))
    freq_ord = freq.sort_index()
    prob_freq = freq_ord.values / freq_ord.values.sum()
    prob_uni = np.full(nmax, 1 / nmax)
    probs = (peso_freq * prob_freq) + ((1 - peso_freq) * prob_uni)

    def maior_seq(jogo):
        jogo = sorted(jogo)
        maior = atual = 1
        for i in range(1, len(jogo)):
            if jogo[i] == jogo[i-1] + 1:
                atual += 1
                maior = max(maior, atual)
            else:
                atual = 1
        return maior

    jogos = []
    vistos = set()
    tentativas = 0

    while len(jogos) < n_jogos and tentativas < 50000:
        tentativas += 1
        escolha = np.random.choice(numeros, size=ndez, replace=False, p=probs)
        candidato = sorted(escolha.tolist())
        chave = tuple(candidato)
        if chave in vistos:
            continue

        soma = sum(candidato)
        pares = sum(1 for n in candidato if n % 2 == 0)
        seq = maior_seq(candidato)

        valido = True
        if faixa_pares and not (faixa_pares[0] <= pares <= faixa_pares[1]):
            valido = False
        if faixa_soma and not (faixa_soma[0] <= soma <= faixa_soma[1]):
            valido = False
        if seq > max_seq:
            valido = False

        if valido:
            vistos.add(chave)
            jogos.append({
                "jogo": candidato,
                "pares": pares,
                "impares": ndez - pares,
                "soma": soma,
                "maior_sequencia": seq
            })

    return pd.DataFrame(jogos)

# =========================
# INTERFACE PRINCIPAL
# =========================

# Sidebar — escolha da loteria e filtros
st.sidebar.header("Configurações")

loteria = st.sidebar.selectbox(
    "Escolha a loteria",
    ["Lotofacil", "Quina", "Mega-Sena"]
)

n_jogos = st.sidebar.slider("Quantidade de jogos", 1, 20, 10)
peso_freq = st.sidebar.slider("Peso da frequência histórica", 0.0, 1.0, 0.7, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros dos jogos**")
usar_filtro_pares = st.sidebar.checkbox("Filtrar por pares/ímpares", value=True)
usar_filtro_soma = st.sidebar.checkbox("Filtrar por soma", value=True)
max_seq = st.sidebar.slider("Sequência máxima permitida", 1, 5, 2)

# Carregar dados
with st.spinner(f"Carregando dados da {loteria}..."):
    df, ndez, nmax = baixar_dados(loteria)

if df is None:
    st.error("Não foi possível carregar os dados. Tente novamente.")
    st.stop()

colunas_dez = [c for c in df.columns if c.startswith("D")]

# Métricas principais
ultimo = df.iloc[-1]
nums_ultimo = sorted(ultimo[colunas_dez].dropna().astype(int).tolist())

col1, col2, col3 = st.columns(3)
col1.metric("Total de concursos", len(df))
col2.metric("Último concurso", int(ultimo["concurso"]))
col3.metric("Data do último sorteio", pd.to_datetime(ultimo["data"]).strftime("%d/%m/%Y") if pd.notna(ultimo["data"]) else "N/A")

st.markdown(f"**Números do último sorteio:** {nums_ultimo}")

st.markdown("---")

# Calcular estatísticas
freq = calcular_freq(df, colunas_dez, nmax)
atrasos = calcular_atrasos(df, colunas_dez, nmax)
pares_serie = calcular_pares(df, colunas_dez, ndez)
soma_serie = calcular_soma(df, colunas_dez)

faixa_soma = (int(soma_serie.quantile(0.20)), int(soma_serie.quantile(0.80)))
faixa_pares_padrao = (
    int(pares_serie.quantile(0.20)),
    int(pares_serie.quantile(0.80))
)

# Abas
aba1, aba2, aba3, aba4 = st.tabs([
    "Frequencia", "Atrasos", "Distribuicoes", "Gerar Jogos"
])

with aba1:
    st.subheader("Frequência das dezenas")
    fig, ax = plt.subplots(figsize=(14, 4))
    freq_sorted = freq.sort_index()
    ax.bar(freq_sorted.index, freq_sorted.values, color="steelblue")
    ax.axhline(freq_sorted.mean(), color="red", linestyle="--",
               label=f"Media: {freq_sorted.mean():.1f}")
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Aparicoes")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 10 mais frequentes**")
        st.dataframe(freq.sort_values(ascending=False).head(10).reset_index())
    with col_b:
        st.markdown("**Top 10 menos frequentes**")
        st.dataframe(freq.sort_values(ascending=True).head(10).reset_index())

with aba2:
    st.subheader("Atraso atual de cada dezena")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    atr_sorted = atrasos.sort_index().dropna()
    ax2.bar(atr_sorted.index, atr_sorted.values, color="mediumpurple")
    ax2.set_xlabel("Dezena")
    ax2.set_ylabel("Concursos sem sair")
    st.pyplot(fig2)
    plt.close()

    st.markdown("**Top 10 dezenas mais atrasadas**")
    st.dataframe(atrasos.sort_values(ascending=False).head(10).reset_index())

with aba3:
    st.subheader("Distribuições históricas")

    col_c, col_d = st.columns(2)

    with col_c:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        pares_count = pares_serie.value_counts().sort_index()
        ax3.bar(pares_count.index, pares_count.values, color="coral", edgecolor="black")
        ax3.set_title("Pares por sorteio")
        ax3.set_xlabel("Quantidade de pares")
        ax3.set_ylabel("Frequencia")
        st.pyplot(fig3)
        plt.close()

    with col_d:
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.hist(soma_serie, bins=35, color="mediumseagreen", edgecolor="black")
        ax4.axvline(soma_serie.mean(), color="red", linestyle="--",
                    label=f"Media: {soma_serie.mean():.1f}")
        ax4.set_title("Soma por sorteio")
        ax4.set_xlabel("Soma")
        ax4.set_ylabel("Frequencia")
        ax4.legend()
        st.pyplot(fig4)
        plt.close()

with aba4:
    st.subheader("Gerador de jogos")

    fp = faixa_pares_padrao if usar_filtro_pares else None
    fs = faixa_soma if usar_filtro_soma else None

    if st.button("Gerar jogos agora"):
        with st.spinner("Gerando jogos..."):
            jogos_df = gerar_jogos(
                freq=freq,
                ndez=ndez,
                nmax=nmax,
                n_jogos=n_jogos,
                peso_freq=peso_freq,
                faixa_pares=fp,
                faixa_soma=fs,
                max_seq=max_seq
            )

        if jogos_df.empty:
            st.warning("Nenhum jogo encontrado. Tente afrouxar os filtros.")
        else:
            st.success(f"{len(jogos_df)} jogos gerados com sucesso!")
            st.dataframe(jogos_df)

            csv = jogos_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Baixar jogos em CSV",
                data=csv,
                file_name=f"jogos_{loteria.lower()}.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Dados obtidos de fontes publicas. Uso estatistico — nao garante acerto.")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(codigo_app)

print("Arquivo app.py criado com sucesso!")
