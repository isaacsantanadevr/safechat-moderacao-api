"""
Camada de classificacao semantica por embeddings (k-NN).

Complementa moderador.py (correspondencia exata via regex) e similaridade.py
(Jaccard de bigramas + distancia de edicao sobre grafia): aquelas camadas
pegam VARIACOES de palavras ja cadastradas em palavras_proibidas.txt. Esta
camada pega mensagens ofensivas que nao usam nenhum termo da lista, mas cujo
SIGNIFICADO e proximo o suficiente de exemplos ja rotulados no dataset
(dados/brutos/mensagens.csv) - insultos com sinonimos, ameacas indiretas,
paranoia diferente da usada nos termos cadastrados.

Abordagem (k-NN sobre embeddings de sentenca):
  1. As mensagens rotuladas do dataset sao convertidas em vetores que
     capturam significado (embeddings_utils.py cuida da geracao/cache).
  2. A mensagem recebida e convertida no mesmo espaco vetorial.
  3. Comparamos por similaridade de cosseno com o dataset inteiro e pegamos
     os K vizinhos mais proximos - eles "votam" numa categoria.
  4. Se a categoria vencedora nao for "normal" e o voto for majoritario o
     suficiente (LIMIAR_CONFIANCA), a mensagem e sinalizada.

Esta camada so roda quando a regex e o Jaccard NAO sinalizaram nada (ver
integracao em moderador.py) - e uma rede de seguranca adicional, nunca
substitui as camadas anteriores, que sao mais baratas e mais precisas para
o que ja esta cadastrado.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from embeddings_utils import carregar_ou_gerar_embeddings, gerar_embeddings

CAMINHO_DATASET = (Path(__file__)).with_name("dados") / "brutos" / "mensagens.csv"
NOME_CACHE_DATASET = "mensagens_rotuladas"

K_VIZINHOS = 5
LIMIAR_CONFIANCA = 0.6  # fracao minima dos k vizinhos que precisa concordar
CATEGORIA_SEGURA = "normal"

MENSAGEM_BLOQUEADA = (
    "[mensagem removida pela moderacao - conteudo sinalizado por analise semantica]"
)

_dataset_cache: dict | None = None


def _carregar_dataset() -> dict:
    """Carrega (e mantem em memoria) os embeddings + categorias do dataset
    rotulado. So roda a leitura/geracao uma vez por processo."""
    global _dataset_cache

    if _dataset_cache is not None:
        return _dataset_cache

    df = pd.read_csv(CAMINHO_DATASET)
    embeddings = carregar_ou_gerar_embeddings(
        NOME_CACHE_DATASET, CAMINHO_DATASET, df["mensagem"].tolist()
    )

    _dataset_cache = {
        "embeddings": embeddings,
        "categorias": df["categoria"].to_numpy(),
        "mensagens": df["mensagem"].tolist(),
    }
    return _dataset_cache


def classificar_por_similaridade(
    mensagem: str,
    k: int = K_VIZINHOS,
    limiar_confianca: float = LIMIAR_CONFIANCA,
) -> tuple[str, float, list[tuple[str, str, float]]]:
    """Classifica `mensagem` comparando-a com o dataset rotulado.

    Retorna (categoria_prevista, confianca, vizinhos). `vizinhos` e a lista
    dos k exemplos mais proximos (mensagem, categoria, similaridade) - util
    para depuracao e para explicar por que uma mensagem foi ou nao sinalizada.
    """
    dataset = _carregar_dataset()
    vetor = gerar_embeddings([mensagem])[0]

    # vetores normalizados -> produto escalar equivale a similaridade de cosseno
    similaridades = dataset["embeddings"] @ vetor
    indices_top_k = np.argsort(-similaridades)[:k]

    categorias_vizinhas = dataset["categorias"][indices_top_k]
    contagem = Counter(categorias_vizinhas)
    categoria_prevista, votos = contagem.most_common(1)[0]
    confianca = votos / k

    vizinhos = [
        (dataset["mensagens"][i], dataset["categorias"][i], float(similaridades[i]))
        for i in indices_top_k
    ]

    if confianca < limiar_confianca:
        return CATEGORIA_SEGURA, confianca, vizinhos

    return categoria_prevista, confianca, vizinhos


def censurar_semantico(mensagem: str) -> str:
    """Camada 3 de moderacao: se o significado da mensagem for proximo o
    bastante de exemplos rotulados como palavrao/insulto/ameaca, ela e
    bloqueada por inteiro.

    Diferente das camadas 1 e 2, aqui nao ha um termo especifico pra
    mascarar caractere a caractere (a ofensa pode estar espalhada pela frase
    inteira, sem nenhuma palavra "culpada"), entao a mensagem inteira e
    substituida por um aviso.
    """
    categoria, _confianca, _vizinhos = classificar_por_similaridade(mensagem)

    if categoria != CATEGORIA_SEGURA:
        return MENSAGEM_BLOQUEADA

    return mensagem


if __name__ == "__main__":
    mensagem_usuario = input("Digite uma mensagem: ")
    categoria, confianca, vizinhos = classificar_por_similaridade(mensagem_usuario)

    print(f"\nCategoria prevista: {categoria} (confianca: {confianca:.0%})")
    print("Vizinhos mais proximos no dataset:")
    for texto, cat, sim in vizinhos:
        print(f"  [{cat}] (sim={sim:.2f}) {texto}")
