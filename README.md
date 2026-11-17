# Modelo de Previsão de Colesterol

Modelo de **regressão linear múltipla** que estima o resultado do exame de colesterol de um paciente a partir de suas características (grupo sanguíneo, tabagismo, nível de atividade física, idade, peso e altura).

O projeto foi desenvolvido para um **laboratório de análises clínicas**: com o modelo, o laboratório tem uma estimativa do nível de colesterol esperado para o perfil do paciente antes do exame. Essa estimativa pode servir para triagem, para priorizar atendimentos e para conferir resultados que fogem muito do esperado.

> ⚠️ O modelo é uma ferramenta de apoio. Ele **não substitui** o exame laboratorial nem a avaliação de um profissional de saúde.

---

## Sumário

- [Fluxo do projeto](#fluxo-do-projeto)
- [Dataset](#dataset)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Etapas](#etapas)
  - [1. Carregar os dados](#1-carregar-os-dados)
  - [2. Análise exploratória (EDA)](#2-análise-exploratória-eda)
  - [3. Preparar datasets de treino e teste](#3-preparar-datasets-de-treino-e-teste)
  - [4. Treinar o modelo](#4-treinar-o-modelo)
  - [5. Validar o modelo](#5-validar-o-modelo)
  - [6. Salvar o modelo](#6-salvar-o-modelo)
  - [7. Entregar o modelo via API](#7-entregar-o-modelo-via-api)
- [Status](#status)

---

## Fluxo do projeto

```
┌──────────────┐   ┌───────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐   ┌─────┐
│ Carregar os  │──▶│  EDA  │──▶│ Split treino /   │──▶│ Treinar  │──▶│ Validar  │──▶│ Salvar │──▶│ API │
│    dados     │   │       │   │ teste + pipeline │   │ modelo   │   │ modelo   │   │ modelo │   │     │
└──────────────┘   └───────┘   └──────────────────┘   └──────────┘   └──────────┘   └────────┘   └─────┘
```

## Dataset

Arquivo: [datasets/cholesterol_dataset.csv](datasets/cholesterol_dataset.csv), com **1.000 registros** de pacientes.

| Coluna original      | Coluna no modelo         | Tipo                | Valores / faixa                 |
| -------------------- | ------------------------ | ------------------- | ------------------------------- |
| `Id`                 | — (descartada)           | identificador       | 1 – 1000                        |
| `Grupo Sanguíneo`    | `grupo_sanguineo`        | categórica nominal  | A, B, AB, O                     |
| `Fumante`            | `fumante`                | categórica nominal  | Sim, Não                        |
| `Nível de Atividade` | `nivel_atividade_fisica` | categórica ordinal  | Baixo < Moderado < Alto         |
| `Idade`              | `idade`                  | numérica (anos)     | 20 – 79                         |
| `Peso`               | `peso`                   | numérica (kg)       | 15,4 – 158,1                    |
| `Altura`             | `altura`                 | numérica (cm)       | 160 – 190                       |
| `Colesterol`         | `nivel_colesterol`       | **alvo** (mg/dL)    | 36,3 – 372,7 (média ≈ 199,5)    |

Há poucos valores ausentes (3 a 4 por coluna) em todas as colunas, exceto `Id` e `Colesterol`.

## Estrutura do repositório

```
cholesterol_model/
├── cholesterol_model.ipynb      # Notebook: carga, EDA, preparação e treino
├── datasets/
│   └── cholesterol_dataset.csv  # Dados dos pacientes
├── Pipfile                      # Dependências (pipenv)
├── Pipfile.lock
└── README.md
```

## Como executar

**Pré-requisitos:** Python 3.15 e [pipenv](https://pipenv.pypa.io/).

```bash
# 1. Instalar as dependências
pipenv install

# 2. Ativar o ambiente virtual
pipenv shell

# 3. Abrir o notebook (VS Code ou Jupyter)
code cholesterol_model.ipynb
```

Principais bibliotecas: `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn` e `gradio`.

---

## Etapas

### 1. Carregar os dados

- Leitura do CSV com `pandas.read_csv`.
- Remoção da coluna `Id`, que não tem valor preditivo.
- Renomeação das colunas para `snake_case`, sem acentos nem espaços.

### 2. Análise exploratória (EDA)

A EDA é feita sobre uma **cópia** do dataset (`df_cholesterol_eda`), para não alterar os dados usados no treino.

- **Valores ausentes:** preenchidos com a **moda** nas variáveis categóricas e com a **mediana** nas numéricas.
- **Tipos:** `idade` e `altura` convertidas para inteiro.
- **Outliers:** pelos boxplots, foram encontrados **37 pacientes com peso < 40 kg**, valor incompatível com a faixa etária do estudo (adultos de 20 a 79 anos). Esses registros são removidos.
- **Distribuições:** boxplots de `idade`, `peso`, `altura` e `nivel_colesterol`.
- **Relação com o alvo:**
  - boxplots de colesterol por `grupo_sanguineo`, `fumante` e `nivel_atividade_fisica`;
  - gráficos de dispersão de colesterol contra `idade`, `peso` e `altura`;
  - `pairplot` com todas as variáveis.
- **Correlação:** as variáveis categóricas são codificadas (one-hot para as nominais, códigos inteiros para a ordinal) e é gerado um heatmap de correlação, com destaque para a correlação de cada variável com `nivel_colesterol`.
- **Bucketing:** agrupamento de `idade` (faixas de 10 anos) e de `peso` (faixas de 10 kg) para ver como o colesterol varia entre as faixas.

### 3. Preparar datasets de treino e teste

- Remoção dos outliers de peso (< 40 kg) no dataset de treino.
- Separação em features (`X`) e alvo (`y = nivel_colesterol`).
- Divisão **70% treino / 30% teste** com `train_test_split(test_size=0.30, random_state=51)`.

O pré-processamento fica dentro de um `ColumnTransformer`. Assim, imputação e codificação são aprendidas **só com os dados de treino**, o que evita vazamento de dados (*data leakage*):

| Grupo        | Colunas                              | Transformações                                                   |
| ------------ | ------------------------------------ | ---------------------------------------------------------------- |
| Numéricas    | `idade`, `altura`, `peso`            | `SimpleImputer(median)` → `StandardScaler`                       |
| Categóricas  | `grupo_sanguineo`, `fumante`         | `SimpleImputer(most_frequent)` → `OneHotEncoder`                 |
| Ordinais     | `nivel_atividade_fisica`             | `SimpleImputer(most_frequent)` → `OrdinalEncoder(Baixo, Moderado, Alto)` |

### 4. Treinar o modelo

O pré-processamento e o regressor ficam em um único `Pipeline` do scikit-learn:

```python
model_regr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression()),
])

model_regr.fit(X_train, y_train)
```

Com tudo no mesmo pipeline, o modelo salvo recebe os **dados brutos do paciente** e aplica sozinho as transformações aprendidas no treino.

### 5. Validar o modelo

#### Métricas

A avaliação é feita no conjunto de **teste**:

| Métrica  | Descrição                                                                 |
| -------- | ------------------------------------------------------------------------- |
| **R²**   | Proporção da variância do colesterol explicada pelo modelo                |
| **MAE**  | Erro absoluto médio, em mg/dL (fácil de interpretar no contexto clínico)  |
| **RMSE** | Raiz do erro quadrático médio, em mg/dL (penaliza mais os erros grandes)  |

```python
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

y_pred = model_regr.predict(X_test)
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
root_mean_squared_error(y_test, y_pred)
```

#### Análise de resíduos

A análise de resíduos (`resíduo = y_real − y_previsto`) verifica se os pressupostos da regressão linear são atendidos:

- **Normalidade:** histograma e gráfico Q-Q dos resíduos, além de teste estatístico (ex.: Shapiro-Wilk ou Kolmogorov-Smirnov, via `scipy.stats`).
- **Homocedasticidade:** dispersão dos resíduos contra os valores previstos. O esperado é uma nuvem sem padrão, com variância constante.
- **Independência / linearidade:** ausência de padrões ou curvaturas nos resíduos.

### 6. Salvar o modelo

O pipeline treinado (pré-processamento + regressor) é salvo em um único arquivo com `joblib`, para ser carregado depois pela API:

```python
import joblib

joblib.dump(model_regr, './model_cholesterol.pkl')
```

### 7. Entregar o modelo via API

O modelo salvo é carregado e exposto por uma interface/API feita com **Gradio**. A equipe do laboratório informa as características do paciente e recebe o nível de colesterol estimado.

Entrada esperada:

```json
{
  "grupo_sanguineo": "A",
  "fumante": "Não",
  "nivel_atividade_fisica": "Moderado",
  "idade": 45,
  "peso": 80.5,
  "altura": 175
}
```

Saída:

```json
{
  "nivel_colesterol": 210.37
}
```

---

## Status

| Etapa                           | Situação         |
| ------------------------------- | ---------------- |
| Carregar os dados               | ✅ Concluída      |
| EDA                             | ✅ Concluída      |
| Preparar datasets treino/teste  | ✅ Concluída      |
| Treinar o modelo                | ✅ Concluída      |
| Validar o modelo (métricas)     | 🚧 Em andamento   |
| Validar o modelo (resíduos)     | 🚧 Em andamento   |
| Salvar o modelo                 | ⏳ Planejada      |
| Entregar modelo via API         | ⏳ Planejada      |
