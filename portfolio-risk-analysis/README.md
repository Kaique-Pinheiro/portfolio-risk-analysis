# Portfolio Risk Analysis B3

> Análise quantitativa de risco de carteira de ações da B3 utilizando metodologias usadas na indústria financeira: VaR, CVaR, Beta de mercado, Simulação de Monte Carlo e Fronteira Eficiente de Markowitz.

---

## Visão Geral

Este projeto implementa um pipeline completo de **risk management quantitativo** voltado para o mercado de renda variável brasileiro. Desenvolvido como portfólio técnico para posições em bancos de investimento (J.P. Morgan, Itaú BBA, Santander), o código demonstra domínio das metodologias exigidas em áreas de **Risk, Quant Finance e Structuring**.

**Carteira analisada:** ITUB4, BBDC4, PETR4, VALE3, BBAS3 (igualmente ponderada, 20% cada)  
**Benchmark:** Ibovespa (^BVSP)  
**Horizonte:** 3 anos de dados históricos via yfinance

---

## Metodologias Implementadas

### Value at Risk (VaR)

Medida de risco que responde: *"Qual é a perda máxima esperada em um dia com X% de confiança?"*

| Método | Premissa | Vantagem | Limitação |
|---|---|---|---|
| **Paramétrico** | Retornos seguem distribuição normal | Analítico, rápido | Subestima fat tails |
| **Histórico** | Distribuição empírica dos retornos passados | Captura assimetria real | Depende da janela histórica |
| **Monte Carlo** | Simulação de N cenários com µ e σ históricos | Flexível, extensível para GBM | Computacionalmente intensivo |

### CVaR / Expected Shortfall (ES)

O **CVaR** (Conditional Value at Risk) é a perda média nos cenários que *excedem* o VaR:

```
CVaR_α = E[L | L > VaR_α]
```

**Por que Basel III prefere CVaR ao VaR?**  
O VaR ignora a magnitude das perdas na cauda — dois portfólios com o mesmo VaR de 2% podem ter CVaRs de 3% e 8%. O CVaR é uma medida de risco coerente (satisfaz subaditividade), o que incentiva diversificação. O framework **FRTB (Basel IV)** substituiu VaR por ES como métrica padrão de capital regulatório.

### Beta de Mercado

O Beta mede a sensibilidade do ativo em relação ao mercado (Ibovespa):

```
β = Cov(R_ativo, R_mercado) / Var(R_mercado)
```

| Beta | Interpretação |
|---|---|
| β > 1 | Ativo mais volátil que o mercado (agressivo) |
| β = 1 | Move exatamente com o mercado |
| 0 < β < 1 | Menos volátil que o mercado (defensivo) |
| β < 0 | Move inversamente ao mercado (hedge natural) |

### Simulação de Monte Carlo

Gera **10.000 cenários** de retorno diário amostrados de uma distribuição normal com parâmetros calibrados nos dados históricos. Permite estimar VaR e CVaR sem premissas analíticas sobre a forma da distribuição.

**Extensão natural:** simulação multivariada com Cholesky decomposition para preservar a estrutura de correlação entre ativos (GBM multivariado).

### Fronteira Eficiente de Markowitz

Baseada na Teoria Moderna do Portfólio (Markowitz, 1952), busca carteiras que **maximizam retorno para dado nível de risco**.

**Sharpe Ratio:**
```
Sharpe = (E[R_p] - Rf) / σ_p
```

Três portfólios são identificados e comparados:

| Portfólio | Critério | Uso |
|---|---|---|
| **Máximo Sharpe** | Maximiza retorno ajustado ao risco | Alocação padrão |
| **Mínima Variância** | Minimiza volatilidade da carteira | Perfil conservador |
| **Igualmente Ponderado** | 20% em cada ativo (baseline) | Benchmark ingênuo |

---

## Estrutura do Projeto

```
portfolio-risk-analysis/
├── analysis.py          <- pipeline principal (único arquivo a executar)
├── requirements.txt     <- dependências pinadas
├── README.md            <- esta documentação
└── .gitignore           <- ignora outputs, __pycache__, .env
```

---

## Instalação e Execução

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/portfolio-risk-analysis.git
cd portfolio-risk-analysis

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar o pipeline
python analysis.py
```

**Requisitos:** Python 3.10+

---

## Dependências

```
yfinance>=0.2.40    # Download de dados históricos da B3
pandas>=2.0         # Manipulação de séries temporais
numpy>=1.26         # Álgebra linear e operações vetorizadas
matplotlib>=3.8     # Visualizações e subplots
scipy>=1.12         # Distribuições estatísticas e otimização SLSQP
```

---

## Saídas do Programa

### Terminal (exemplo de saída)

```
=================================================================
[1] DOWNLOAD DE DADOS
=================================================================
  Ativos      : ['ITUB4.SA', 'BBDC4.SA', 'PETR4.SA', 'VALE3.SA', 'BBAS3.SA']
  Benchmark   : ^BVSP
  Shape       : (754, 5) (dias x ativos)
  Período     : 22/04/2023 -> 22/04/2026
  Dados baixados com sucesso!

=================================================================
[2] RETORNOS E ESTATÍSTICAS DESCRITIVAS
=================================================================
       Retorno Anual (%)  Volatilidade Anual (%)  Skewness  Kurtosis  Sharpe Ratio
ITUB4             18.42                   28.61    -0.2341    3.1204        0.2981
...
```

### Gráficos gerados (`risk_analysis.png`)

| Subplot | Título | Descrição |
|---|---|---|
| 1 (topo) | Retorno Acumulado | Evolução dos 5 ativos em base 1.0 |
| 2 | Distribuição + VaR | Histograma com curva normal e 3 linhas de VaR |
| 3 | Heatmap de Correlação | Matriz de correlação com valores anotados |
| 4 | VaR por Ativo | Bar chart comparativo do VaR histórico (95%) |
| 5 | Monte Carlo | Histograma dos 10k cenários simulados |
| 6 (base) | Fronteira Eficiente | Scatter de 3.000 carteiras colorido por Sharpe |

---

## Ativos Analisados

| Ticker | Empresa | Setor |
|---|---|---|
| ITUB4.SA | Itaú Unibanco | Financeiro — Bancos |
| BBDC4.SA | Banco Bradesco | Financeiro — Bancos |
| PETR4.SA | Petrobras | Energia — Petróleo e Gás |
| VALE3.SA | Vale | Materiais — Mineração |
| BBAS3.SA | Banco do Brasil | Financeiro — Bancos |
| ^BVSP | Ibovespa | Benchmark de mercado |

**Taxa livre de risco:** Selic ≈ 10,75% a.a. (referência para cálculo do Sharpe e VaR paramétrico)

---

## Conceitos Financeiros Chave

### Por que retornos logarítmicos?

```python
# Retorno simples (não-aditivo no tempo):
ret_simples = (P_t / P_{t-1}) - 1

# Retorno logarítmico (aditivo no tempo, propriedade essencial para séries temporais):
ret_log = np.log(P_t / P_{t-1})
```

Retornos logarítmicos são preferidos porque: (1) são aditivos temporalmente, (2) assumem valores em (-∞, +∞) sem restrição de -100%, (3) são mais compatíveis com a hipótese de normalidade.

### Por que anualizar com √252?

```python
# 252 = número médio de dias de negociação por ano na B3
volatilidade_anual = volatilidade_diaria * np.sqrt(252)
retorno_anual      = retorno_diario_medio * 252
```

Decorre da propriedade de que, se retornos diários são i.i.d., a variância anual é 252× a variância diária, portanto o desvio-padrão anual é √252 × o desvio-padrão diário.

### Por que comparar VaR Paramétrico vs Histórico?

Mercados financeiros exibem **fat tails** — eventos extremos ocorrem com frequência maior do que a distribuição normal prevê (fenômeno documentado por Mandelbrot e Taleb). Quando VaR Histórico > VaR Paramétrico, os dados confirmam que as caudas reais são mais pesadas que a normal — validação empírica da escolha pelo CVaR como métrica regulatória.

---

## Próximos Passos

- [ ] **Backtesting de VaR** — testes de Kupiec (POF) e Christoffersen (independência das violações)
- [ ] **GBM Multivariado** — Monte Carlo com Cholesky decomposition para preservar correlações
- [ ] **Stress Testing** — cenários históricos (crise 2008, COVID-19, eleições BR) e hipotéticos
- [ ] **Streamlit Dashboard** — interface interativa para seleção dinâmica de ativos e pesos
- [ ] **Otimização com restrições** — turnover máximo, exposição setorial, drawdown máximo

---

## Referências

- **Markowitz, H.** (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- **J.P. Morgan / Reuters** (1996). *RiskMetrics — Technical Document* (4th ed.).
- **Hull, J.C.** (2018). *Risk Management and Financial Institutions* (5th ed.). Wiley.
- **Basel Committee on Banking Supervision** (2019). *Minimum capital requirements for market risk (FRTB)*.

---

## Autor

**Kaique** — Estudante de Engenharia, FEI  
GitHub: [github.com/seu-usuario](https://github.com/seu-usuario)  
Email: seu-email@fei.edu.br
