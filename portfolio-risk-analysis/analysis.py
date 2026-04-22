"""
Portfolio Risk Analysis — B3
Pipeline completo de análise quantitativa de risco de carteira de ações brasileiras.
Metodologias: VaR, CVaR, Beta, Monte Carlo, Backtesting de Kupiec e Fronteira Eficiente de Markowitz.
"""

# Suprime avisos de deprecação e warnings internos das bibliotecas
# para deixar a saída do terminal limpa
import warnings
warnings.filterwarnings('ignore')

# numpy  → operações matemáticas e vetorizadas (arrays, álgebra linear)
# pandas → manipulação de tabelas e séries temporais (DataFrames)
# matplotlib → criação dos gráficos
# gridspec → layout de múltiplos subplots em uma figura
# FuncFormatter → formatar os eixos dos gráficos como porcentagem
# yfinance → baixar dados históricos de ações diretamente do Yahoo Finance
# scipy.stats → distribuições estatísticas (normal, skewness, etc.)
# scipy.optimize.minimize → otimização numérica para a fronteira de Markowitz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize

# ─────────────────────────────────────────────
# CONSTANTES GLOBAIS
# Centralizadas aqui para facilitar ajustes sem
# precisar alterar o código dentro das funções.
# ─────────────────────────────────────────────

# Tickers das ações no formato do Yahoo Finance (".SA" = Bolsa brasileira B3)
TICKERS       = ['ITUB4.SA', 'BBDC4.SA', 'PETR4.SA', 'VALE3.SA', 'BBAS3.SA']

# Benchmark: Ibovespa — índice de referência do mercado brasileiro
BENCHMARK     = '^BVSP'

# Período histórico para download: '3y' = últimos 3 anos
PERIOD        = '3y'

# Nível de confiança do VaR/CVaR: 0.95 significa que cobrimos 95% dos cenários,
# ou seja, medimos a perda nos piores 5% dos dias
CONF          = 0.95

# Taxa livre de risco (Selic anualizada ≈ 10,75%) usada no cálculo do Sharpe Ratio:
# Sharpe = (Retorno da carteira - Selic) / Volatilidade
RISK_FREE     = 0.1075

# Número de cenários gerados na simulação de Monte Carlo
N_SIM         = 10_000

# Vetor de pesos da carteira igualmente ponderada: 20% em cada um dos 5 ativos
# np.array converte a lista Python em um array numérico para operações matriciais
WEIGHTS_EQUAL = np.array([0.20] * 5)

# Cores corporativas para os gráficos (hexadecimal):
# azul escuro (Itaú), dourado (Bradesco), vermelho (Petrobras), verde (Vale), laranja (BB)
COLORS = ['#003087', '#d4af37', '#e63946', '#2a9d8f', '#e76f51']
LABELS = ['Itaú Unibanco', 'Bradesco', 'Petrobras', 'Vale', 'Banco do Brasil']

# Estilo visual dos gráficos: fundo escuro com grade, padrão "seaborn darkgrid"
plt.style.use('seaborn-v0_8-darkgrid')


# ═══════════════════════════════════════════════════════════════════════════════
# [1] DOWNLOAD DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

def baixar_dados():
    """Faz download dos preços históricos via yfinance para todos os ativos."""
    print("=" * 65)
    print("[1] DOWNLOAD DE DADOS")
    print("=" * 65)

    # Junta os 5 tickers da carteira com o benchmark numa lista só
    # para fazer um único download em lote (mais eficiente)
    todos_tickers = TICKERS + [BENCHMARK]

    # Tentativa principal: baixar tudo de uma vez com yf.download()
    # auto_adjust=True já ajusta os preços por dividendos e desdobramentos (splits),
    # então não precisamos fazer esse ajuste manualmente depois
    try:
        raw = yf.download(todos_tickers, period=PERIOD, auto_adjust=True)
        # raw é um DataFrame com múltiplos níveis de colunas (Open, High, Low, Close, Volume)
        # Selecionamos apenas 'Close' — preço de fechamento de cada dia
        precos_raw = raw['Close']
    except Exception:
        # Se o download em lote falhar (ex: erro de rede), inicializa vazio
        # e tenta baixar individualmente no bloco abaixo
        precos_raw = pd.DataFrame()

    # Verifica se algum ticker ficou sem dados (coluna ausente ou tudo NaN)
    # Isso acontece quando o cache local do yfinance está bloqueado por outro processo
    faltando = [t for t in todos_tickers if t not in precos_raw.columns
                or precos_raw[t].isna().all()]

    # Para cada ticker que falhou, faz download individual como fallback
    for ticker in faltando:
        try:
            single = yf.download(ticker, period=PERIOD, auto_adjust=True)['Close']
            precos_raw[ticker] = single  # adiciona a coluna que estava faltando
        except Exception as e:
            print(f"  AVISO: nao foi possivel baixar {ticker}: {e}")

    # dropna() remove qualquer linha onde pelo menos um ativo não tem preço naquele dia
    # Isso garante que todos os ativos têm dados no mesmo conjunto de datas
    precos = precos_raw[todos_tickers].dropna()

    # Proteção: se o DataFrame ficou completamente vazio, aborta com mensagem clara
    if precos.empty:
        raise RuntimeError("Nenhum dado disponivel apos download. Verifique conexao.")

    # Separa os preços em dois DataFrames distintos para uso nas etapas seguintes
    precos_ativos = precos[TICKERS]    # DataFrame 5 colunas (um por ativo)
    precos_bvsp   = precos[BENCHMARK]  # Series com preço diário do Ibovespa

    # Formata as datas do índice para exibição legível no terminal
    data_inicio = precos.index[0].strftime('%d/%m/%Y')
    data_fim    = precos.index[-1].strftime('%d/%m/%Y')

    print(f"  Ativos      : {TICKERS}")
    print(f"  Benchmark   : {BENCHMARK}")
    print(f"  Shape       : {precos_ativos.shape} (dias x ativos)")
    print(f"  Periodo     : {data_inicio} -> {data_fim}")
    print("  Dados baixados com sucesso!\n")

    return precos_ativos, precos_bvsp


# ═══════════════════════════════════════════════════════════════════════════════
# [2] RETORNOS E ESTATÍSTICAS DESCRITIVAS
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_estatisticas(precos_ativos, precos_bvsp):
    """Calcula retornos logarítmicos e estatísticas descritivas anualizadas."""
    print("=" * 65)
    print("[2] RETORNOS E ESTATÍSTICAS DESCRITIVAS")
    print("=" * 65)

    # Retorno logarítmico diário: ln(P_t / P_{t-1})
    # Preferimos log-retorno ao retorno simples porque:
    # 1) É aditivo no tempo: ret_semana = soma dos ret_diários
    # 2) Aproxima melhor a distribuição normal, facilitando os modelos estatísticos
    # shift(1) desloca a série 1 dia para trás (P_{t-1}), permitindo a divisão
    # dropna() remove a primeira linha, que fica NaN porque não há P_{t-1}
    retornos      = np.log(precos_ativos / precos_ativos.shift(1)).dropna()
    retornos_bvsp = np.log(precos_bvsp   / precos_bvsp.shift(1)).dropna()

    # Garante que ativos e benchmark têm exatamente as mesmas datas
    # (às vezes o Ibovespa tem um dia a mais/menos por calendário)
    idx_comum     = retornos.index.intersection(retornos_bvsp.index)
    retornos      = retornos.loc[idx_comum]
    retornos_bvsp = retornos_bvsp.loc[idx_comum]

    # Calcula as estatísticas para cada ativo e armazena num dicionário
    stats_data = {}
    for ticker in TICKERS:
        r  = retornos[ticker]  # série de retornos diários deste ativo
        mu = r.mean()          # média aritmética dos retornos diários
        sg = r.std()           # desvio-padrão dos retornos diários

        stats_data[ticker] = {
            # Anualizando o retorno: multiplicamos por 252 (dias úteis/ano na B3)
            # e por 100 para converter de decimal para porcentagem
            'Retorno Anual (%)':      round(mu * 252 * 100, 2),

            # Anualizando a volatilidade: pela regra da raiz do tempo,
            # se retornos diários são independentes, σ_anual = σ_diário × √252
            'Volatilidade Anual (%)': round(sg * np.sqrt(252) * 100, 2),

            # Skewness (assimetria): mede se a distribuição pende para a direita (+)
            # ou para a esquerda (-). Valor negativo = mais dias de queda acentuada
            'Skewness':               round(stats.skew(r), 4),

            # Kurtosis (curtose): mede a "gordura" das caudas da distribuição
            # Distribuição normal tem curtose 0 (excess kurtosis).
            # Valor > 0 indica fat tails — eventos extremos ocorrem mais que o esperado
            'Kurtosis':               round(r.kurt(), 4),

            # Sharpe Ratio: retorno acima da taxa livre de risco por unidade de risco
            # Fórmula: (Retorno_anual - Selic) / Volatilidade_anual
            # Quanto maior, melhor o ativo remunera o risco assumido
            'Sharpe Ratio':           round((mu * 252 - RISK_FREE) / (sg * np.sqrt(252)), 4),
        }

    # Converte o dicionário em DataFrame e transpõe (T) para ter ativos nas linhas
    df_stats = pd.DataFrame(stats_data).T
    # Remove o sufixo ".SA" dos índices para ficar mais limpo na exibição
    df_stats.index = [t.replace('.SA', '') for t in df_stats.index]

    print(df_stats.to_string())
    print()

    return retornos, retornos_bvsp


# ═══════════════════════════════════════════════════════════════════════════════
# [3] VaR, CVaR E BETA
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_var_cvar_beta(retornos, retornos_bvsp):
    """Calcula VaR (3 métodos), CVaR e Beta de mercado por ativo."""
    print("=" * 65)
    print("[3] VaR, CVaR E BETA")
    print("=" * 65)

    # Retorno diário da carteira: multiplicação matricial dos retornos × pesos
    # Operador @ é produto matricial em Python. Resultado: uma série com o retorno
    # diário da carteira inteira (já ponderada 20% em cada ativo)
    port_ret = retornos @ WEIGHTS_EQUAL

    mu_port    = port_ret.mean()  # média diária do retorno da carteira
    sigma_port = port_ret.std()   # desvio-padrão diário (= volatilidade diária)

    # ── VaR PARAMÉTRICO (método analítico, assume normalidade)
    # norm.ppf(p) retorna o quantil p da distribuição normal padrão
    # Para CONF=0.95: ppf(0.05) ≈ -1.645 (o "z-score" do pior 5%)
    z         = stats.norm.ppf(1 - CONF)
    # Fórmula: VaR = -(µ + z × σ)
    # O sinal negativo converte perda (negativa) em número positivo para comparação
    var_param = -(mu_port + z * sigma_port)

    # ── VaR HISTÓRICO (método não-paramétrico, usa a distribuição real dos dados)
    # np.percentile extrai o percentil (1-CONF)×100 = 5º percentil da série
    # Ou seja: 5% dos dias tiveram retorno abaixo deste valor
    var_hist  = -np.percentile(port_ret, (1 - CONF) * 100)

    # ── CVaR / Expected Shortfall (ES) HISTÓRICO
    # CVaR = média dos retornos nos dias que foram PIORES que o VaR
    # Responde: "quando o VaR é superado, qual é a perda média?"
    # Basel III/IV e FRTB exigem CVaR porque o VaR não diz nada sobre o tamanho
    # da perda nos piores dias — o CVaR captura esse risco de cauda (tail risk)
    cvar_hist = -port_ret[port_ret <= -var_hist].mean()

    print(f"  Carteira    : Igualmente ponderada (20% por ativo)")
    print(f"  Confiança   : {CONF*100:.0f}%\n")
    print(f"  VaR Paramétrico  : {var_param*100:.4f}%")
    print(f"  VaR Histórico    : {var_hist*100:.4f}%")
    print(f"  CVaR (ES) Hist.  : {cvar_hist*100:.4f}%\n")

    # ── BETA DE MERCADO por ativo
    # Beta mede quanto o ativo oscila em relação ao mercado (Ibovespa)
    # Fórmula do CAPM: β = Cov(R_ativo, R_mercado) / Var(R_mercado)
    # β > 1: ativo mais volátil que o mercado (amplifica movimentos)
    # β < 1: ativo mais defensivo (atenua movimentos)
    var_bvsp = retornos_bvsp.var()  # variância do Ibovespa (denominador)
    betas = {}
    for ticker in TICKERS:
        # np.cov retorna uma matriz 2×2; posição [0,1] é a covariância cruzada
        cov_ativo = np.cov(retornos[ticker].values, retornos_bvsp.values)[0, 1]
        betas[ticker.replace('.SA', '')] = round(cov_ativo / var_bvsp, 4)

    # Converte o dicionário de betas em DataFrame para exibição tabular
    df_betas = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta vs IBOV'])
    print("  Betas de mercado:")
    print(df_betas.to_string())
    print()

    return port_ret, var_param, var_hist, cvar_hist, betas


# ═══════════════════════════════════════════════════════════════════════════════
# [4] SIMULAÇÃO DE MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo(port_ret):
    """Simula N_SIM cenários de retorno diário e calcula VaR/CVaR por simulação."""
    print("=" * 65)
    print("[4] SIMULAÇÃO DE MONTE CARLO")
    print("=" * 65)

    mu_port    = port_ret.mean()  # média histórica diária da carteira
    sigma_port = port_ret.std()   # desvio-padrão histórico diário

    # Fixamos a semente aleatória para reprodutibilidade:
    # rodar o script duas vezes sempre produz os mesmos 10.000 números
    np.random.seed(42)

    # Gera N_SIM números aleatórios com distribuição normal N(µ, σ)
    # Representa 10.000 possíveis retornos diários da carteira
    # É o modelo mais simples de Monte Carlo — extensão natural seria GBM multivariado
    simulados = np.random.normal(mu_port, sigma_port, N_SIM)

    # Aplica as mesmas fórmulas de VaR/CVaR, agora nos dados simulados
    var_mc  = -np.percentile(simulados, (1 - CONF) * 100)
    cvar_mc = -simulados[simulados <= -var_mc].mean()

    print(f"  Cenários simulados : {N_SIM:,}")
    print(f"  mu (diário)        : {mu_port*100:.4f}%")
    print(f"  sigma (diário)     : {sigma_port*100:.4f}%")
    print(f"  VaR MC ({CONF*100:.0f}%)    : {var_mc*100:.4f}%")
    print(f"  CVaR MC ({CONF*100:.0f}%)   : {cvar_mc*100:.4f}%\n")

    return simulados, var_mc, cvar_mc


# ═══════════════════════════════════════════════════════════════════════════════
# [5] BACKTESTING DE KUPIEC (Proportion of Failures — POF Test)
# ═══════════════════════════════════════════════════════════════════════════════

def backtesting_kupiec(port_ret, var_param, var_hist, var_mc):
    """
    Testa se a taxa de violações observada é estatisticamente compatível
    com o nível de confiança declarado — metodologia exigida pelo Basel II/III.

    H0: o modelo está bem calibrado (taxa de violações = 1 - CONF)
    H1: o modelo subestima ou superestima o risco

    A estatística LR segue distribuição qui-quadrado com 1 grau de liberdade.
    """
    print("=" * 65)
    print("[5] BACKTESTING DE KUPIEC (POF Test)")
    print("=" * 65)
    print(f"  H0: taxa de violacoes = {(1-CONF)*100:.0f}% (modelo bem calibrado)")
    print(f"  Critico qui2(1) a 5% de significancia = 3.841\n")

    T = len(port_ret)          # número total de dias observados
    p = 1 - CONF               # taxa de violação esperada pelo modelo (5%)

    def _kupiec_test(var_value, nome):
        """Executa o POF test de Kupiec para um único valor de VaR."""

        # Uma violação ocorre quando a perda real (- retorno) supera o VaR previsto
        # port_ret < -var_value é equivalente a: perda_real > VaR_estimado
        violacoes_mask = port_ret < -var_value   # Series booleana, True = dia ruim
        N              = int(violacoes_mask.sum())  # contagem total de violações
        p_hat          = N / T                    # taxa de violação observada

        # Casos extremos: log(0) é indefinido — modelo trivialmente errado
        if N == 0 or N == T:
            print(f"  [{nome}]  N={N} — estatistica indefinida (log(0))\n")
            return {'nome': nome, 'var': var_value, 'N': N, 'T': T,
                    'p_hat': p_hat, 'lr': np.nan, 'p_value': np.nan,
                    'rejeita': None, 'violacoes_mask': violacoes_mask}

        # ── Estatística Log-Likelihood Ratio de Kupiec:
        # LR = -2 × [ ln L(p) − ln L(p_hat) ]
        # onde L(p) é a verossimilhança sob H0 e L(p_hat) sob H1 (irrestrita)
        # Distribuição binomial: L(q) = (1-q)^(T-N) × q^N
        lr = -2 * (
            (T - N) * np.log(1 - p)     + N * np.log(p)        # ln L(p)  — H0
          - (T - N) * np.log(1 - p_hat) - N * np.log(p_hat)    # ln L(p_hat) — H1
        )

        # p-value: probabilidade de observar LR >= valor calculado sob H0
        # stats.chi2.cdf(lr, df=1) = P(X <= lr) → 1 - isso = P(X >= lr) = p-value
        p_val   = 1 - stats.chi2.cdf(lr, df=1)
        rejeita = p_val < 0.05   # rejeita H0 se p-value < nível de significância

        status = "REPROVADO [X]" if rejeita else "APROVADO  [OK]"
        print(f"  [{nome}]")
        print(f"    Observacoes (T)   : {T} dias")
        print(f"    VaR estimado      : {var_value*100:.4f}%")
        print(f"    Violacoes (N)     : {N}  ({p_hat*100:.2f}% dos dias)")
        print(f"    Taxa esperada (p) : {p*100:.2f}%")
        print(f"    Estatistica LR    : {lr:.4f}  (critico = 3.841)")
        print(f"    p-value           : {p_val:.4f}")
        print(f"    Resultado         : {status}")
        print()

        return {
            'nome':           nome,
            'var':            var_value,
            'N':              N,
            'T':              T,
            'p_hat':          p_hat,
            'lr':             lr,
            'p_value':        p_val,
            'rejeita':        rejeita,
            'violacoes_mask': violacoes_mask,   # usado no gráfico de timeline
        }

    # Executa o teste para os 3 métodos de VaR
    resultados = {
        'parametrico': _kupiec_test(var_param, 'VaR Parametrico'),
        'historico':   _kupiec_test(var_hist,  'VaR Historico'),
        'monte_carlo': _kupiec_test(var_mc,    'VaR Monte Carlo'),
    }
    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# [6] FRONTEIRA EFICIENTE DE MARKOWITZ
# ═══════════════════════════════════════════════════════════════════════════════

def fronteira_eficiente(retornos):
    """Gera fronteira eficiente com 3.000 carteiras aleatórias e otimização SLSQP."""
    print("=" * 65)
    print("[6] FRONTEIRA EFICIENTE DE MARKOWITZ")
    print("=" * 65)

    n = len(TICKERS)  # número de ativos (5)

    # Calcula o retorno médio anualizado de cada ativo (vetor µ)
    mu_anual = retornos.mean() * 252

    # Calcula a matriz de covariância anualizada (matriz Σ de 5×5)
    # A covariância captura não só a variância individual, mas também
    # como os ativos se movem juntos — base da diversificação de Markowitz
    cov_anual  = retornos.cov() * 252
    cov_matrix = cov_anual.values  # converte para array numpy puro (mais rápido)

    # Função auxiliar que, dado um vetor de pesos w, retorna (retorno, volatilidade, sharpe)
    def portfolio_stats(w):
        # Retorno esperado da carteira: média ponderada dos retornos individuais
        ret = np.dot(w, mu_anual.values)

        # Volatilidade da carteira: fórmula matricial √(wᵀ Σ w)
        # Captura o efeito da correlação entre ativos — diversificação reduz σ
        vol = np.sqrt(w @ cov_matrix @ w)

        # Sharpe Ratio: retorno excedente (acima da Selic) por unidade de risco
        sharpe = (ret - RISK_FREE) / vol
        return ret, vol, sharpe

    # ── ETAPA 1: Gerar 3.000 carteiras aleatórias para visualizar o espaço risco-retorno
    carteiras = {'ret': [], 'vol': [], 'sharpe': [], 'weights': []}
    for _ in range(3000):
        # Distribuição Dirichlet com parâmetro α=1 gera vetores aleatórios que
        # somam exatamente 1.0 — garante que os pesos são uma alocação válida
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_stats(w)
        carteiras['ret'].append(r)
        carteiras['vol'].append(v)
        carteiras['sharpe'].append(s)
        carteiras['weights'].append(w)

    # ── ETAPA 2: Otimização numérica com SLSQP (Sequential Least Squares Programming)
    # SLSQP é eficiente para problemas com restrições de igualdade e desigualdade

    # Restrição: a soma dos pesos deve ser exatamente 1 (100% do capital alocado)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: cada ativo pode receber entre 0% e 100% (sem posições vendidas)
    bounds = tuple((0, 1) for _ in range(n))

    # Ponto inicial da otimização: carteira igualmente ponderada (20% cada)
    w0 = np.ones(n) / n

    # Portfólio de Máximo Sharpe: minimizamos o negativo do Sharpe
    # (scipy só tem minimizador, então maximizar Sharpe = minimizar -Sharpe)
    res_sharpe   = minimize(lambda w: -portfolio_stats(w)[2],
                            w0, method='SLSQP', bounds=bounds, constraints=constraints)
    w_max_sharpe = res_sharpe.x  # vetor de pesos ótimos
    ret_ms, vol_ms, sr_ms = portfolio_stats(w_max_sharpe)

    # Portfólio de Mínima Variância: minimizamos diretamente a volatilidade
    res_minvol = minimize(lambda w: portfolio_stats(w)[1],
                          w0, method='SLSQP', bounds=bounds, constraints=constraints)
    w_min_vol  = res_minvol.x
    ret_mv, vol_mv, sr_mv = portfolio_stats(w_min_vol)

    # Portfólio igualmente ponderado: serve como benchmark de comparação
    ret_eq, vol_eq, sr_eq = portfolio_stats(WEIGHTS_EQUAL)

    # Remove sufixo ".SA" para exibição limpa
    tickers_curtos = [t.replace('.SA', '') for t in TICKERS]

    print("  Portfolio de Maximo Sharpe:")
    for t, w in zip(tickers_curtos, w_max_sharpe):
        print(f"    {t}: {w*100:.1f}%")
    print(f"  -> Retorno: {ret_ms*100:.2f}% | Vol: {vol_ms*100:.2f}% | Sharpe: {sr_ms:.4f}\n")

    print("  Portfolio de Minima Variancia:")
    for t, w in zip(tickers_curtos, w_min_vol):
        print(f"    {t}: {w*100:.1f}%")
    print(f"  -> Retorno: {ret_mv*100:.2f}% | Vol: {vol_mv*100:.2f}% | Sharpe: {sr_mv:.4f}\n")

    # Retorna tudo que os gráficos precisarão
    return {
        'carteiras':    carteiras,
        'w_max_sharpe': w_max_sharpe,
        'w_min_vol':    w_min_vol,
        'stats': {
            'max_sharpe': (ret_ms, vol_ms, sr_ms),
            'min_vol':    (ret_mv, vol_mv, sr_mv),
            'equal':      (ret_eq, vol_eq, sr_eq),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# [7] VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

def gerar_graficos(precos_ativos, retornos, port_ret,
                   var_param, var_hist, var_mc,
                   simulados, otimos, resultados_kupiec):
    """Gera figura com 7 subplots e salva como risk_analysis.png."""
    print("=" * 65)
    print("[7] GERANDO VISUALIZACOES")
    print("=" * 65)

    # FuncFormatter transforma números decimais em strings de porcentagem nos eixos
    # Ex: 0.015 → "1.5%" nos rótulos do gráfico
    fmt_pct = FuncFormatter(lambda x, _: f'{x*100:.1f}%')

    # Cria a figura principal com tamanho 18×22 polegadas e fundo branco
    fig = plt.figure(figsize=(18, 22), facecolor='white')
    fig.suptitle('Portfolio Risk Analysis — B3', fontsize=20, fontweight='bold', y=0.98)

    # GridSpec define o layout de 5 linhas × 2 colunas para os 7 gráficos
    # hspace = espaço vertical entre subplots | wspace = espaço horizontal
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ─────────────────────────────────────────────────────────────
    # GRÁFICO 1: Retorno Acumulado
    # gs[0, :] significa linha 0, todas as colunas (ocupa a linha inteira)
    # ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    # Converte log-retornos de volta para retornos simples: e^(log-ret) - 1
    # Necessário porque o produto acumulado só funciona com retornos simples
    retornos_simples = np.exp(retornos) - 1

    # cumprod() calcula o produto acumulado: cada dia multiplica o total anterior
    # Ex: dias +1%, -0.5%, +2% → 1.01 × 0.995 × 1.02 = valor acumulado
    acumulado = (1 + retornos_simples).cumprod()

    for i, ticker in enumerate(TICKERS):
        ax1.plot(acumulado.index, acumulado[ticker],
                 color=COLORS[i], label=LABELS[i], linewidth=2)

    # Linha horizontal em 1.0 = linha de base (sem ganho nem perda)
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}x'))
    ax1.set_title('Retorno Acumulado (base 1.0)', fontsize=13, fontweight='bold')
    ax1.legend(ncol=5, fontsize=9)
    ax1.set_xlabel('Data')

    # ─────────────────────────────────────────────────────────────
    # GRÁFICO 2: Distribuição de retornos + 3 linhas de VaR
    # Permite comparar visualmente os 3 métodos e ver as fat tails
    # ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    # Histograma dos retornos históricos da carteira (density=True normaliza a área para 1)
    ax2.hist(port_ret, bins=60, density=True, color='#003087', alpha=0.6,
             label='Retornos hist.')

    # Curva da distribuição normal teórica com os mesmos µ e σ dos dados reais
    # Serve para visualizar o quanto os dados reais se desviam da normal (fat tails)
    x_range = np.linspace(port_ret.min(), port_ret.max(), 300)
    ax2.plot(x_range,
             stats.norm.pdf(x_range, port_ret.mean(), port_ret.std()),
             color='#d4af37', linewidth=2, label='Normal ajustada')

    # Três linhas verticais mostrando onde cada VaR cai na distribuição
    # O VaR fica na cauda esquerda (perdas grandes = valores negativos)
    ax2.axvline(-var_param, color='#e63946', linewidth=2, linestyle='--',
                label=f'VaR Param. {var_param*100:.2f}%')
    ax2.axvline(-var_hist,  color='#2a9d8f', linewidth=2, linestyle='-.',
                label=f'VaR Hist. {var_hist*100:.2f}%')
    ax2.axvline(-var_mc,    color='#e76f51', linewidth=2, linestyle=':',
                label=f'VaR MC {var_mc*100:.2f}%')

    ax2.xaxis.set_major_formatter(fmt_pct)
    ax2.set_title('Distribuicao de Retornos + VaR (95%)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.set_xlabel('Retorno Diario')

    # ─────────────────────────────────────────────────────────────
    # GRÁFICO 3: Heatmap de Correlação
    # Mostra como os ativos se movem juntos — base da diversificação
    # ─────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    # Calcula a matriz de correlação de Pearson entre os retornos dos ativos
    # Valores variam de -1 (movimentos opostos) a +1 (movimentos idênticos)
    corr = retornos.corr()
    tickers_curtos = [t.replace('.SA', '') for t in TICKERS]

    # imshow renderiza a matriz como imagem colorida
    # cmap='RdYlGn': vermelho (correlação alta) → amarelo (neutra) → verde (baixa)
    im = ax3.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)  # barra de escala lateral

    # Configura os rótulos dos eixos com os nomes dos tickers
    ax3.set_xticks(range(len(tickers_curtos)))
    ax3.set_yticks(range(len(tickers_curtos)))
    ax3.set_xticklabels(tickers_curtos, fontsize=9)
    ax3.set_yticklabels(tickers_curtos, fontsize=9)

    # Anota o valor numérico dentro de cada célula da matriz
    # Cor do texto: preto se a célula for clara, branco se for escura
    for i in range(len(tickers_curtos)):
        for j in range(len(tickers_curtos)):
            ax3.text(j, i, f'{corr.iloc[i, j]:.2f}',
                     ha='center', va='center', fontsize=9,
                     color='black' if abs(corr.iloc[i, j]) < 0.7 else 'white')
    ax3.set_title('Heatmap de Correlacao', fontsize=11, fontweight='bold')

    # ─────────────────────────────────────────────────────────────
    # GRÁFICO 4: VaR Histórico por ativo (gráfico de barras)
    # Compara o risco individual de cada ação da carteira
    # ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])

    # Calcula o VaR histórico de cada ativo individualmente (não da carteira)
    # e converte para percentagem multiplicando por 100
    vars_ativos = [-np.percentile(retornos[t], (1 - CONF) * 100) * 100 for t in TICKERS]

    bars = ax4.bar(tickers_curtos, vars_ativos, color=COLORS, edgecolor='white', linewidth=0.8)

    # Anota o valor percentual acima de cada barra
    for bar, val in zip(bars, vars_ativos):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_title('VaR Historico (95%) por Ativo', fontsize=11, fontweight='bold')
    ax4.set_ylabel('VaR (%)')
    ax4.set_ylim(0, max(vars_ativos) * 1.25)  # 25% de margem no topo para os rótulos

    # ─────────────────────────────────────────────────────────────
    # GRÁFICO 5: Histograma dos 10.000 cenários de Monte Carlo
    # ─────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])

    ax5.hist(simulados, bins=80, density=True, color='#2a9d8f', alpha=0.7,
             label=f'{N_SIM:,} cenarios')

    # Linha vertical marcando onde o VaR Monte Carlo cai na distribuição simulada
    ax5.axvline(-var_mc, color='#e63946', linewidth=2.5, linestyle='--',
                label=f'VaR MC {var_mc*100:.2f}%')

    ax5.xaxis.set_major_formatter(fmt_pct)
    ax5.set_title(f'Simulacao de Monte Carlo ({N_SIM:,} cenarios)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.set_xlabel('Retorno Diario Simulado')

    # ─────────────────────────────────────────────────────────────
    # GRÁFICO 6: Fronteira Eficiente de Markowitz
    # gs[3, :] = linha 3, ambas as colunas (ocupa a linha inteira)
    # ─────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    cart = otimos['carteiras']

    # Scatter das 3.000 carteiras aleatórias: eixo X = risco (vol), Y = retorno
    # A cor de cada ponto representa o Sharpe Ratio (cmap 'plasma': roxo→amarelo)
    # A "nuvem" de pontos tem a forma característica da fronteira eficiente
    sc = ax6.scatter(cart['vol'], cart['ret'],
                     c=cart['sharpe'], cmap='plasma', alpha=0.5, s=12,
                     label='Carteiras aleatorias')
    plt.colorbar(sc, ax=ax6, label='Sharpe Ratio')  # barra de cores à direita

    # Extrai as estatísticas dos 3 portfólios ótimos para plotar como pontos destacados
    ret_ms, vol_ms, _ = otimos['stats']['max_sharpe']
    ret_mv, vol_mv, _ = otimos['stats']['min_vol']
    ret_eq, vol_eq, _ = otimos['stats']['equal']

    # Portfólio igualmente ponderado: diamante dourado
    ax6.scatter(vol_eq, ret_eq, color='#d4af37', s=180, zorder=5,
                marker='D', label='Carteira Igual (20% cada)', edgecolors='black')

    # Portfólio de Máximo Sharpe: estrela azul (zorder=5 garante que fica na frente)
    ax6.scatter(vol_ms, ret_ms, color='#003087', s=250, zorder=5,
                marker='*', label='Maximo Sharpe', edgecolors='black')

    # Portfólio de Mínima Variância: cruz vermelha (mais à esquerda = menos risco)
    ax6.scatter(vol_mv, ret_mv, color='#e63946', s=180, zorder=5,
                marker='P', label='Minima Variancia', edgecolors='black')

    ax6.xaxis.set_major_formatter(fmt_pct)
    ax6.yaxis.set_major_formatter(fmt_pct)
    ax6.set_title('Fronteira Eficiente de Markowitz', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Volatilidade Anual')
    ax6.set_ylabel('Retorno Anual')
    ax6.legend(fontsize=9)

    # ─────────────────────────────────────────────────────────────
    # GRÁFICO 7: Timeline de violações do Backtesting de Kupiec
    # gs[4, :] = linha 4, ambas as colunas (ocupa a linha inteira)
    # Mostra quando o VaR foi violado ao longo do tempo — visualmente
    # fácil de identificar períodos de estresse (COVID, crises, eleições)
    # ─────────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, :])

    # Perdas diárias da carteira: nega os retornos para que dias de queda fiquem positivos
    # Isso facilita comparar visualmente com o VaR (que é sempre positivo)
    perdas = -port_ret

    # Barras cinzas representam cada dia — altura = perda (se positivo) ou ganho (negativo)
    ax7.bar(perdas.index, perdas, color='#cccccc', linewidth=0, label='Perda/Ganho diario')

    # Linhas horizontais indicando o nível de cada VaR
    # Dias com barra acima da linha = violação
    kup_param = resultados_kupiec['parametrico']
    kup_hist  = resultados_kupiec['historico']
    ax7.axhline(var_param, color='#e63946', linewidth=1.5, linestyle='--',
                label=f"VaR Param. {var_param*100:.2f}% (N={kup_param['N']})")
    ax7.axhline(var_hist, color='#2a9d8f', linewidth=1.5, linestyle='-.',
                label=f"VaR Hist. {var_hist*100:.2f}% (N={kup_hist['N']})")

    # Pontos vermelhos marcam os dias em que o VaR Histórico foi violado
    # violacoes_mask é a Series booleana calculada na função backtesting_kupiec
    viol_idx = kup_hist['violacoes_mask']
    ax7.scatter(perdas.index[viol_idx], perdas[viol_idx],
                color='#e63946', s=30, zorder=5, label='Violacao VaR Hist.')

    ax7.yaxis.set_major_formatter(fmt_pct)
    ax7.set_title(
        f"Backtesting de Kupiec — Violacoes do VaR ao Longo do Tempo"
        f"  |  Aprovado: {'Sim' if not kup_hist['rejeita'] else 'Nao'}"
        f"  |  p-value: {kup_hist['p_value']:.4f}",
        fontsize=11, fontweight='bold'
    )
    ax7.set_xlabel('Data')
    ax7.set_ylabel('Perda/Ganho Diario')
    ax7.legend(fontsize=9, ncol=4)

    # Salva a figura completa como PNG no diretório atual do script
    # dpi=150 define a resolução (150 pixels por polegada = boa qualidade)
    # bbox_inches='tight' corta margens em branco ao redor da figura
    plt.savefig('risk_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("  Figura salva: risk_analysis.png (7 subplots)\n")
    plt.close()  # libera memória fechando a figura


# ═══════════════════════════════════════════════════════════════════════════════
# [8] RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def relatorio_final(port_ret, var_param, var_hist, var_mc,
                    cvar_hist, cvar_mc, resultados_kupiec):
    """Exibe resumo executivo com interpretação financeira dos resultados."""
    print("=" * 65)
    print("[8] RELATORIO FINAL — RESUMO EXECUTIVO")
    print("=" * 65)

    # f-string multilinha: interpola as variáveis calculadas nas etapas anteriores
    # var_hist * 100_000 simula o impacto em reais para uma carteira de R$ 100 mil
    print(f"""
  CARTEIRA ANALISADA
  ------------------------------------------------------------------
  Ativos  : ITUB4, BBDC4, PETR4, VALE3, BBAS3 (B3)
  Pesos   : 20% cada (igualmente ponderada)
  Periodo : {PERIOD} | Confianca: {CONF*100:.0f}% | Selic: {RISK_FREE*100:.2f}%

  METRICAS DE RISCO DIARIO
  ------------------------------------------------------------------
  VaR Parametrico : {var_param*100:>8.4f}%   (assume distribuicao normal)
  VaR Historico   : {var_hist*100:>8.4f}%   (baseado em dados empiricos)
  VaR Monte Carlo : {var_mc*100:>8.4f}%   (10.000 cenarios simulados)
  CVaR Historico  : {cvar_hist*100:>8.4f}%   (perda media nos piores dias)
  CVaR Monte Carlo: {cvar_mc*100:>8.4f}%   (perda media nos cenarios MC)

  BACKTESTING DE KUPIEC (validacao estatistica dos modelos)
  ------------------------------------------------------------------"""
    )

    # Exibe uma linha por método de VaR com o veredito do teste
    for chave, r in resultados_kupiec.items():
        if r['rejeita'] is None:
            veredito = "INDEFINIDO"
        elif r['rejeita']:
            veredito = "REPROVADO — modelo subestima o risco"
        else:
            veredito = "APROVADO  — taxa de violacoes compativel"
        print(f"  {r['nome']:<20}: N={r['N']}/{r['T']}  "
              f"p-value={r['p_value']:.4f}  -> {veredito}")

    print(f"""
  INTERPRETACAO
  ------------------------------------------------------------------
  -> Com {CONF*100:.0f}% de confianca, a perda maxima diaria esperada
     e de {var_hist*100:.2f}% (VaR Historico).
     Para uma carteira de R$ 100.000, a perda de um dia ruim
     (5% dos piores dias) ultrapassa R$ {var_hist*100_000:.0f}.

  -> O CVaR de {cvar_hist*100:.2f}% representa a perda MEDIA nos dias
     que superam o limiar do VaR — captura o risco de cauda
     (tail risk) ignorado pelo VaR simples.
     Basel III/IV e FRTB exigem CVaR (ES) como medida padrao.

  -> Kupiec POF Test: um modelo APROVADO significa que a frequencia
     real de violacoes nao diverge significativamente de {(1-CONF)*100:.0f}%
     (p-value > 0.05 = nao rejeitamos H0). Modelos reprovados
     devem ser recalibrados antes de uso em capital regulatorio.
    """)
    print("=" * 65)
    print("  Pipeline concluido com sucesso!")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# O bloco `if __name__ == '__main__'` garante que o pipeline só executa
# quando o arquivo é rodado diretamente (python analysis.py),
# e não quando importado como módulo por outro script
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Etapa 1: baixa os preços históricos do Yahoo Finance
    precos_ativos, precos_bvsp = baixar_dados()

    # Etapa 2: transforma preços em retornos logarítmicos e calcula estatísticas
    retornos, retornos_bvsp = calcular_estatisticas(precos_ativos, precos_bvsp)

    # Etapa 3: calcula VaR (3 métodos), CVaR e Beta de cada ativo
    port_ret, var_param, var_hist, cvar_hist, betas = calcular_var_cvar_beta(
        retornos, retornos_bvsp
    )

    # Etapa 4: simulação de Monte Carlo com 10.000 cenários
    simulados, var_mc, cvar_mc = monte_carlo(port_ret)

    # Etapa 5: backtesting de Kupiec — valida os 3 modelos de VaR estatisticamente
    resultados_kupiec = backtesting_kupiec(port_ret, var_param, var_hist, var_mc)

    # Etapa 6: fronteira eficiente — 3.000 carteiras aleatórias + otimização SLSQP
    otimos = fronteira_eficiente(retornos)

    # Etapa 7: gera e salva os 7 gráficos em risk_analysis.png
    gerar_graficos(
        precos_ativos, retornos, port_ret,
        var_param, var_hist, var_mc,
        simulados, otimos, resultados_kupiec
    )

    # Etapa 8: imprime o resumo executivo com interpretação dos resultados
    relatorio_final(port_ret, var_param, var_hist, var_mc,
                    cvar_hist, cvar_mc, resultados_kupiec)
