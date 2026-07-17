# CensuraBot

Serviço de moderação usado pelo SafeChat. A API recebe uma mensagem, aplica
três camadas em ordem de custo e devolve o texto censurado.

## Fluxo de moderação

1. **Bag of Words:** correspondência exata com `palavras_proibidas.txt`.
2. **Jaccard + Levenshtein:** encontra leetspeak, letras repetidas e palavras
   separadas por símbolos.
3. **Embeddings TF-IDF + k-NN:** compara a mensagem completa com o dataset
   rotulado. O modelo é ajustado somente no primeiro uso e mantido em memória.
   Essa camada usa apenas exemplos sem termo ofensivo explícito e adota um
   limiar conservador para não censurar conversas cotidianas por aproximação.

Se uma das duas primeiras camadas censurar algum trecho, a terceira não é
executada. Isso mantém o caminho de termos conhecidos simples e rápido.

## Execução

```powershell
python -m pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Contrato consumido pelo SafeChat:

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

O endpoint `GET /health` retorna `{"status": "ok"}`.

## Ferramentas offline

Os scripts de descoberta de termos e detecção de quase duplicatas continuam
podendo usar embeddings Transformer, sem afetar a latência da API:

```powershell
python -m pip install -r requirements-offline.txt
python descobrir_termos.py
python src/validar_dados.py
```

## Testes

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
