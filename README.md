# 🔍 Análise Exploratória de Dados - RH

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Status](https://img.shields.io/badge/status-finalizado-brightgreen)]()

Este projeto realiza uma análise exploratória e pré-processamento de um conjunto de dados do setor de Recursos Humanos (RH), com foco na identificação de profissionais interessados em mudar de emprego. O trabalho foi desenvolvido utilizando Python e bibliotecas como Pandas, Matplotlib, Seaborn, Scikit-learn e Plotly.

---

## 📁 Sobre o Dataset

O conjunto de dados utilizado (`aug_train.csv`) contém informações sobre candidatos, incluindo:

- Localização (cidade e índice de desenvolvimento)
- Nível de educação e área de formação
- Tipo e tamanho da empresa atual
- Experiência e horas de treinamento
- Se estão ou não procurando novas oportunidades (variável `target`)

---

## ⚙️ Etapas do Projeto

### ✅ Importação e Visualização Inicial
- Leitura dos dados e estruturação do DataFrame
- Identificação de colunas, tipos e valores nulos

### 📊 Análise Exploratória (EDA)
- Gráficos de contagem para variáveis categóricas
- Histogramas e boxplots para variáveis numéricas
- Análise de distribuição e normalidade
- Verificação de desbalanceamento da variável alvo

### 🔗 Correlação e Variáveis Relevantes
- Cálculo da correlação Spearman entre variáveis numéricas
- Heatmap das correlações
- Cálculo de WOE (Weight of Evidence) e IV (Information Value) para variáveis categóricas

### 🧹 Tratamento de Dados Ausentes
- Visualização com `missingno`
- Estratégias condicionais para preenchimento com base em outras variáveis
- Substituição por categorias como `"Non Degree"`, `"Other"` ou `"Primary Grad"`

### ✂️ Ajustes Finais
- Conversão de categorias especiais (`<1`, `>20`) para valores numéricos
- Normalização de nomenclatura (`no_enrollment` → `No enrollment`)
- Remoção de valores faltantes restantes
- Separação de features (`X`) e target (`y`)

---

## 📦 Bibliotecas Utilizadas

- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly`
- `missingno`
- `scipy.stats`
- `sklearn.preprocessing`, `sklearn.impute`, `sklearn.pipeline`
- `category_encoders`

---

## 📈 Visualização de Exemplo

![Gráfico Exemplo](https://i.postimg.cc/jjbVFKx9/Heatmap.png)

---

## 📌 Objetivo Final

Este projeto prepara os dados para modelagem preditiva, com o intuito de **prever a probabilidade de um candidato estar buscando uma nova oportunidade de trabalho**, com base em seu perfil educacional, profissional e demográfico.

---

## 📈 Possíveis Extensões

- Treinamento de modelos de Machine Learning (Logistic Regression, Random Forest, XGBoost)
- Feature Selection e Engenharia de Atributos
- Deploy com Streamlit ou Flask

---

## 🧠 Autor

**Mateus Philipe Valverde de Salles**  
Estagiário de TI, estudante e entusiasta em Ciência de Dados e Análise Exploratória.  
[🔗 GitHub](https://github.com/Omateuso) | 📧 E-mail mateusphilipe.valverde@hotmail.com
