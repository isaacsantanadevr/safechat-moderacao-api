<div align="center">

# 🧠 SafeChat Moderation API

**Moderação automática de linguagem ofensiva em três camadas — regex, similaridade textual e classificação semântica**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20k--NN-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-CensuraBot-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)

</div>

---

Serviço de moderação automática de linguagem ofensiva, consumido pelo
chat [`safechat`](https://github.com/isaacsantanadevr/safechat). Recebe uma
mensagem de texto, aplica três camadas de detecção em ordem crescente de
custo computacional, e devolve o texto (censurado ou não).

> 🔗 Este serviço está rodando em produção, integrado ao chat publicado em
> **[projeto-pln-safechat.up.railway.app](https://projeto-pln-safechat.up.railway.app/)**.
> Não tem interface própria para navegador — é consumido internamente pelo
> chat via `MODERATION_API_URL` (ver *Contrato da API*, abaixo).

## 🧱 Arquitetura em três camadas

```
mensagem
   │
   ▼
1) Bag of Words (regex exato)          — palavras_proibidas.txt
   │  (se nada foi encontrado)
   ▼
2) Jaccard + Levenshtein                — similaridade.py
   │  (se nada foi encontrado)
   ▼
3) TF-IDF + k-NN (similaridade semântica) — classificador_semantico.py
   │
   ▼
resultado final
```

**Camada 1 — Bag of Words (`moderador.py`)**
Um único regex, montado a partir de `palavras_proibidas.txt` (198 termos
atualmente), com fronteira de palavra para não confundir "cu" dentro de
"cultura", por exemplo. Rápida e determinística.

**Camada 2 — Jaccard + Levenshtein (`similaridade.py`)**
Cobre disfarces que a correspondência exata não pega: leetspeak
(`p0rra`, `id10ta`), letras repetidas (`caraaalho`), palavras separadas
por pontuação ou espaço (`c-a-r-a-l-h-o`, `b o s t a`) e acentos usados
como disfarce. Usa bigramas de caracteres (índice de Jaccard) e distância
de edição de Levenshtein normalizada, com um limiar calibrado para não
gerar falso positivo (lista `PALAVRAS_PERMITIDAS` protege palavras reais
que colidem por acaso, como "sábado" perto de "safado").

**Camada 3 — TF-IDF + k-NN (`classificador_semantico.py`)**
Só executa se as duas primeiras camadas não encontraram nada. Vetoriza a
mensagem inteira (combinando n-gramas de palavras e de caracteres via
`FeatureUnion`) e compara por similaridade de cosseno com os exemplos do
dataset rotulado, usando os 5 vizinhos mais próximos com voto ponderado
pela força da similaridade. Só é treinada com mensagens do dataset que
**não têm** um termo ofensivo explícito — essas já são resolvidas pelas
camadas 1 e 2, e incluí-las aqui faria o modelo associar palavras neutras
de contexto (ex.: "acabou") ao palavrão vizinho no exemplo de treino.

Se a camada 2 ou 3 falhar por qualquer motivo (exceção inesperada), o
`moderador.py` **não propaga o erro** — a mensagem segue com o resultado
das camadas anteriores, sem derrubar a API.

## 🧰 Stack tecnológico

| Ferramenta | Papel |
|---|---|
| FastAPI | API REST (`api.py`) |
| Streamlit | interface manual de teste, "CensuraBot" (`app.py`) |
| Pandas / NumPy | manipulação do dataset e dos vetores |
| scikit-learn | TF-IDF, k-NN (implementado à mão) e agrupamento (`KMeans`) |
| joblib | cache em disco do vetorizador TF-IDF usado pelas ferramentas offline |
| pytest | suíte de testes automatizados |
| matplotlib | gráfico da matriz de confusão |

## 📁 Estrutura do projeto

```
api.py                        # API FastAPI: POST /moderate, GET /health
app.py                        # interface Streamlit manual (CensuraBot)
moderador.py                  # orquestra as 3 camadas, em ordem
similaridade.py                # camada 2: Jaccard + Levenshtein + normalização de disfarces
classificador_semantico.py    # camada 3: TF-IDF + k-NN ponderado
embeddings_utils.py            # TF-IDF com cache em disco, usado pelas ferramentas OFFLINE
descobrir_termos.py            # ferramenta manual: sugere novos termos por agrupamento semântico
avaliar_classificador.py       # avaliação formal: treino/teste, métricas, matriz de confusão
src/validar_dados.py           # valida o CSV e detecta quase-duplicatas semânticas
palavras_proibidas.txt         # lista de termos da camada 1 (198 termos)
dados/brutos/mensagens.csv     # dataset rotulado (681 mensagens, 4 categorias)
tests/
  ├── test_moderador.py             # testes de integração das 3 camadas juntas
  └── test_classificador_tfidf.py   # testes unitários da camada 3
requirements.txt               # dependências de produção (API + Streamlit)
requirements-dev.txt            # + pytest
requirements-offline.txt        # + sentence-transformers (ver observação abaixo)
```

## ⚙️ Configurando o ambiente

Requer **Python 3.10+** (testado com 3.12).

```bash
python -m venv .venv
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```
**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Instale as dependências de produção:
```bash
pip install -r requirements.txt
```

> **Nota sobre `requirements-offline.txt`:** esse arquivo inclui
> `sentence-transformers`, pensado originalmente para as ferramentas
> offline (`descobrir_termos.py` e `src/validar_dados.py`) usarem
> embeddings de linguagem em vez de TF-IDF. **Isso não é mais necessário
> hoje** — as duas ferramentas usam `embeddings_utils.py`, que já é só
> TF-IDF (scikit-learn), igual à camada 3 da API. Testamos rodando as duas
> ferramentas **sem** `sentence-transformers` instalado e ambas funcionam
> normalmente. Você pode instalar `requirements-offline.txt` se quiser
> (não quebra nada), mas não é preciso para nenhuma funcionalidade atual
> do projeto — `requirements.txt` já é suficiente para tudo.

## ▶️ Como rodar a API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Teste com `curl` (ou abra `http://localhost:8000/docs` para a interface
Swagger interativa, gerada automaticamente pelo FastAPI):

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "Voce e um idiota"}'
# {"original_content":"Voce e um idiota","content":"Voce e um ******","moderated":true}

curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "Bom dia pessoal"}'
# {"original_content":"Bom dia pessoal","content":"Bom dia pessoal","moderated":false}

curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "Eu sei onde voce mora"}'
# {"original_content":"Eu sei onde voce mora","content":"[mensagem removida pela moderacao - conteudo sinalizado por analise semantica]","moderated":true}
```

O terceiro exemplo não contém nenhuma palavra da lista proibida — é a
camada 3 (semântica) que reconhece a ameaça pelo sentido da frase.

É esta API que o repositório [`safechat`](https://github.com/isaacsantanadevr/safechat)
consome (variável de ambiente `MODERATION_API_URL`, padrão
`http://localhost:8000`).

## 💬 Como rodar a interface manual (CensuraBot)

```bash
streamlit run app.py
```

Abre em `http://localhost:8501`. Digite uma mensagem e clique em
"Analisar" para ver o resultado das três camadas sem precisar montar uma
requisição HTTP.

## ⌨️ Como rodar o moderador pela linha de comando

```bash
python moderador.py
```

Pede uma mensagem pelo terminal e imprime o resultado censurado — útil
para testar rapidamente sem subir a API.

## 📊 Como rodar a avaliação formal (matriz de confusão e métricas)

```bash
python avaliar_classificador.py
```

O que o script faz:

1. Carrega `dados/brutos/mensagens.csv` e seleciona só as mensagens **sem**
   termo ofensivo explícito (a fatia que a camada 3 realmente precisa
   resolver — 306 das 681 mensagens atualmente).
2. Divide esses 306 registros em treino (80%) e teste (20%), de forma
   estratificada por categoria, com semente fixa (`random_state=42`) para
   o resultado ser reproduzível.
3. Ajusta o vetorizador TF-IDF **somente no treino** (evita vazamento de
   dados: o teste nunca "vaza" para o vocabulário aprendido).
4. Classifica cada mensagem de teste com o mesmo algoritmo de produção
   (k-NN ponderado, `k=5`) e compara com a categoria real.
5. Imprime um relatório de precisão/revocação/F1 por categoria e salva
   `matriz_confusao.png` na raiz do projeto.

Saída obtida rodando com o dataset atual (681 mensagens, 306 sem termo
explícito → 244 treino / 62 teste):

```
              precision    recall  f1-score   support

      normal       0.86      1.00      0.92        49
     insulto       0.00      0.00      0.00         2
      ameaca       1.00      0.45      0.62        11

    accuracy                           0.87        62
   macro avg       0.62      0.48      0.52        62
weighted avg       0.86      0.87      0.84        62
```

**Leitura dos números:**
- **Acurácia geral: 87%** nas 62 mensagens de teste.
- **Normal**: revocação de 100% (nenhuma mensagem normal foi confundida
  com ofensiva), mas precisão de 86% — algumas mensagens ofensivas "sem
  termo explícito" acabam classificadas como normais (erro para o lado
  seguro, não o contrário).
- **Insulto**: suporte de apenas 2 exemplos nesta fatia específica do
  dataset — a métrica (0.00 em tudo) não é estatisticamente conclusiva
  com uma amostra dessa quantidade; é uma limitação a citar honestamente,
  não um bug.
- **Ameaça**: revocação de 45% — a categoria mais difícil para uma
  técnica lexical como TF-IDF, já que depende mais do sentido da frase
  inteira do que de palavras-chave isoladas.

Esses números mudam se o dataset for alterado (mais exemplos, categorias
rebalanceadas). Rode o script de novo depois de qualquer mudança em
`dados/brutos/mensagens.csv` para atualizar o relatório e o gráfico.

## 🧪 Como rodar os testes automatizados

```bash
pip install -r requirements-dev.txt
pytest -q
```

Resultado esperado: **28 testes, todos passando** (`28 passed`). Pode
aparecer um aviso (`SyntaxWarning: invalid escape sequence '\w'`) vindo de
uma docstring em `moderador.py` — é só um aviso cosmético do
interpretador, não afeta o funcionamento nem indica falha em teste
nenhum.

O que cada arquivo de teste cobre:

- **`tests/test_moderador.py`** — testes de ponta a ponta através de
  `censurar_mensagem()`, cobrindo as três camadas juntas: mensagens
  normais preservadas (inclusive uma lista de frases do dia a dia que
  poderiam gerar falso positivo, como "eu te amo" e "cuidado na escada"),
  termo exato pego pela camada 1, variação de grafia pega pela camada 2,
  e ameaças sem palavra proibida pegas pela camada 3.
- **`tests/test_classificador_tfidf.py`** — testes unitários da camada 3
  isoladamente: cache do classificador em memória, classificação correta
  de uma ameaça implícita, mensagem sem relação com o dataset
  classificada como segura, similaridade sempre entre 0 e 1, e validação
  de parâmetro (`k` inválido levanta erro).

## 🔍 Ferramentas offline (uso manual, fora da API)

Estas duas ferramentas não são chamadas por `api.py` nem por
`moderador.py` — são utilitários que os desenvolvedores rodam manualmente
para analisar e manter o dataset.

### `descobrir_termos.py`

```bash
python descobrir_termos.py
```

Agrupa as mensagens já rotuladas como ofensivas (palavrão/insulto/ameaça)
por similaridade semântica (`KMeans`, 6 grupos) e lista, por grupo, as
palavras mais frequentes que **ainda não estão** em
`palavras_proibidas.txt`. Não edita o arquivo sozinho — só sugere
candidatos para revisão humana.

### `src/validar_dados.py`

```bash
python src/validar_dados.py
```

Faz duas coisas:

1. **Validação estrutural**: confere se as colunas são exatamente
   `mensagem`, `categoria`, `termos_ofensivos`, se não há mensagem vazia,
   se toda categoria é uma das 4 válidas, e reporta duplicatas exatas.
2. **Detecção de quase-duplicatas semânticas**: usa similaridade de
   cosseno entre embeddings (limiar 0.92) para achar pares de mensagens
   parafraseadas — texto diferente, mesmo sentido — que a checagem de
   duplicata exata não pega. Isso importa porque a camada 3 usa esse CSV
   como base de comparação (k-NN): uma mensagem quase-duplicada "pesa"
   duas vezes entre os vizinhos mais próximos.

Rodando com o dataset atual: **0 duplicatas exatas**, e **cerca de 110
pares de quase-duplicatas** acima do limiar de 0.92 (a maioria são
variações de pontuação do mesmo exemplo-molde, como "Porra, acabou a
bateria" vs. "Porra... acabou a bateria"). A ferramenta só relata os
pares — decidir se vale a pena remover cada um é uma revisão manual dos
desenvolvedores, não uma ação automática.

## 📚 Dataset atual

`dados/brutos/mensagens.csv` — **681 mensagens rotuladas**, sem rotina
automática de coleta (dataset estático, montado e curado pelos
desenvolvedores):

| Categoria | Registros |
|---|---|
| Normal | 240 |
| Palavrão | 220 |
| Insulto | 167 |
| Ameaça | 54 |
| **Total** | **681** |

Destas, **306 mensagens não têm um termo ofensivo explícito** cadastrado
em `termos_ofensivos` — são as usadas para treinar e avaliar a camada 3
(veja *Como rodar a avaliação formal*, acima).

## 📡 Contrato da API

```http
POST /moderate
Content-Type: application/json

{"content": "mensagem enviada pelo usuário"}
```

Resposta:
```json
{
  "original_content": "mensagem enviada pelo usuário",
  "content": "mensagem após a moderação",
  "moderated": false
}
```

`GET /health` retorna `{"status": "ok"}`.

## ⚠️ Limitações conhecidas

- **Tratamento de exceções silencioso**: as camadas 2 e 3 são protegidas
  por blocos `try/except` genéricos, sem `logging`, para que uma falha
  inesperada não derrube a API — mas isso também significa que um erro
  real ali pode passar despercebido nos logs.
- **Cobertura da camada semântica**: a camada 3 só enxerga os 306
  registros sem termo explícito; frases muito diferentes desse conjunto
  tendem a ser classificadas como "normal" por segurança (limiar de
  similaridade mínima conservador).
- **Execução sequencial rígida**: a camada 3 só roda se as camadas 1 e 2
  não encontraram nada — uma mensagem com um termo óbvio e, na mesma
  frase, uma ameaça mais sutil, não chega a passar pela análise
  semântica.

## 🔗 Repositório do chat

Este serviço é consumido pelo chat em tempo real
[`safechat`](https://github.com/isaacsantanadevr/safechat) (Spring Boot +
WebSocket), que tem seu próprio README com instruções de execução.

## 👥 Desenvolvedores

- [@isaacsantanadevr](https://github.com/isaacsantanadevr)
- [@gabrielmarcone](https://github.com/gabrielmarcone)
- [@joaoguilhermedss](https://github.com/joaoguilhermedss)
- [@ccaiomatos](https://github.com/ccaiomatos)

Projeto desenvolvido para a disciplina de Processamento de Linguagem
Natural — UESB.
