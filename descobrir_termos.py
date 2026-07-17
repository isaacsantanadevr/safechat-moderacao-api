"""
Ferramenta OFFLINE de analise - roda manualmente (`python descobrir_termos.py`),
fora do fluxo da API. Nao e chamada pelo moderador.py nem pela api.py.

Agrupa (clusteriza) as mensagens ja rotuladas como palavrao/insulto/ameaca por
similaridade SEMANTICA e, dentro de cada grupo, lista as palavras mais
frequentes que AINDA NAO estao em palavras_proibidas.txt.

Por que isso e util: cada cluster tende a reunir mensagens com o mesmo "tipo"
de ofensa (ex: um cluster de xingamentos com uma palavra, outro de ameacas de
violencia, outro de insultos sobre aparencia). As palavras mais frequentes de
um cluster que ainda nao estao cadastradas sao candidatas fortes a gírias ou
variacoes novas para adicionar na lista - sem precisar ler mensagem por
mensagem manualmente.

Importante: isto sugere CANDIDATOS. A decisao de adicionar um termo à lista
continua sendo humana - o script nao edita palavras_proibidas.txt sozinho.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from embeddings_utils import carregar_ou_gerar_embeddings
from moderador import cargar_termos_proibidos

CAMINHO_DATASET = (Path(__file__)).with_name("dados") / "brutos" / "mensagens.csv"
CATEGORIAS_OFENSIVAS = {"palavrao", "insulto", "ameaca"}

N_CLUSTERS = 6
TOP_PALAVRAS_POR_CLUSTER = 8
TAMANHO_MINIMO_PALAVRA = 3

_PADRAO_PALAVRA = re.compile(r"[a-zA-ZÀ-ÿ]{" + str(TAMANHO_MINIMO_PALAVRA) + r",}")

# palavras comuns do portugues que nao interessam como "termo ofensivo novo",
# mesmo aparecendo com frequencia dentro de um cluster
PALAVRAS_IGNORADAS = {
    "que", "para", "com", "uma", "seu", "sua", "voce", "isso", "esse", "essa",
    "nao", "mais", "muito", "esta", "das", "dos", "por", "como", "ser", "vou",
    "tem", "foi", "sao", "mas", "ele", "ela", "meu", "minha", "nos", "aqui",
}


def _tokenizar(mensagem: str) -> list[str]:
    return [
        palavra.lower()
        for palavra in _PADRAO_PALAVRA.findall(mensagem)
        if palavra.lower() not in PALAVRAS_IGNORADAS
    ]


def descobrir_termos_candidatos(
    n_clusters: int = N_CLUSTERS,
) -> dict[int, list[tuple[str, int]]]:
    """Retorna, por cluster, as palavras mais frequentes que ainda nao estao
    cadastradas em palavras_proibidas.txt."""
    df = pd.read_csv(CAMINHO_DATASET)
    df_ofensivo = df[df["categoria"].isin(CATEGORIAS_OFENSIVAS)].reset_index(drop=True)

    if len(df_ofensivo) < n_clusters:
        raise ValueError(
            "Poucas mensagens ofensivas no dataset para formar "
            f"{n_clusters} clusters (encontradas: {len(df_ofensivo)})."
        )

    mensagens = df_ofensivo["mensagem"].tolist()
    embeddings = carregar_ou_gerar_embeddings(
        "mensagens_ofensivas", CAMINHO_DATASET, mensagens
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rotulos_cluster = kmeans.fit_predict(embeddings)

    termos_conhecidos = set(cargar_termos_proibidos())

    resultado: dict[int, list[tuple[str, int]]] = {}
    for cluster_id in range(n_clusters):
        mensagens_do_cluster = [
            mensagens[i] for i, c in enumerate(rotulos_cluster) if c == cluster_id
        ]

        contador: Counter[str] = Counter()
        for mensagem in mensagens_do_cluster:
            for palavra in _tokenizar(mensagem):
                if palavra not in termos_conhecidos:
                    contador[palavra] += 1

        resultado[cluster_id] = contador.most_common(TOP_PALAVRAS_POR_CLUSTER)

    return resultado


if __name__ == "__main__":
    candidatos_por_cluster = descobrir_termos_candidatos()

    print("Candidatos a novos termos para palavras_proibidas.txt, por cluster semantico:\n")
    for cluster_id, palavras in candidatos_por_cluster.items():
        if not palavras:
            continue
        print(f"Cluster {cluster_id}:")
        for palavra, frequencia in palavras:
            print(f"  {palavra} ({frequencia}x)")
        print()
