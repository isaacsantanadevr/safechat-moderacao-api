import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))

from embeddings_utils import carregar_ou_gerar_embeddings  # noqa: E402 (precisa do sys.path acima)

CAMINHO_DADOS = (
    Path(__file__).resolve().parent.parent
    / "dados"
    / "brutos"
    / "mensagens.csv"
)

COLUNAS_ESPERADAS = {
    "mensagem",
    "categoria",
    "termos_ofensivos",
}

CATEGORIAS_VALIDAS = {
    "normal",
    "palavrao",
    "insulto",
    "ameaca",
}


def carregar_dados() -> pd.DataFrame:
    if not CAMINHO_DADOS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {CAMINHO_DADOS}"
        )

    return pd.read_csv(CAMINHO_DADOS)


def validar_dados(df: pd.DataFrame) -> None:
    colunas_encontradas = set(df.columns)

    if colunas_encontradas != COLUNAS_ESPERADAS:
        raise ValueError(
            f"Colunas incorretas: {list(df.columns)}"
        )

    if df["mensagem"].isna().any():
        raise ValueError("Existem mensagens vazias.")

    categorias_invalidas = (
        set(df["categoria"].unique())
        - CATEGORIAS_VALIDAS
    )

    if categorias_invalidas:
        raise ValueError(
            f"Categorias inválidas: {categorias_invalidas}"
        )

    duplicadas = df["mensagem"].duplicated().sum()

    print(f"Total de mensagens: {len(df)}")
    print(f"Mensagens duplicadas: {duplicadas}")
    print("\nQuantidade por categoria:")
    print(df["categoria"].value_counts())


def detectar_quase_duplicatas(
    df: pd.DataFrame,
    limiar: float = 0.92,
) -> list[tuple[str, str, float]]:
    """Alem da duplicata EXATA ja checada em validar_dados (mesma string),
    acha pares de mensagens PARAFRASEADAS - textos diferentes mas com o
    mesmo significado - usando similaridade de cosseno entre embeddings.

    Isso importa porque o classificador_semantico.py usa este CSV como base
    de comparacao (k-NN): quase-duplicatas nao atrapalham a classificacao,
    mas indicam redundancia no dataset (o mesmo caso "pesando" varias vezes
    nos vizinhos mais proximos) que vale limpar antes de expandir o dataset.

    Retorna uma lista de (mensagem_a, mensagem_b, similaridade), ordenada da
    mais similar para a menos similar.
    """
    mensagens = df["mensagem"].tolist()
    embeddings = carregar_ou_gerar_embeddings(
        "mensagens_rotuladas", CAMINHO_DADOS, mensagens
    )

    matriz_similaridade = embeddings @ embeddings.T
    n = len(mensagens)
    pares_encontrados = []

    for i in range(n):
        for j in range(i + 1, n):
            similaridade = float(matriz_similaridade[i, j])
            if similaridade >= limiar:
                pares_encontrados.append((mensagens[i], mensagens[j], similaridade))

    pares_encontrados.sort(key=lambda par: par[2], reverse=True)
    return pares_encontrados


if __name__ == "__main__":
    dados = carregar_dados()
    validar_dados(dados)

    print("\nProcurando quase-duplicatas (similaridade semantica >= 0.92)...")
    quase_duplicatas = detectar_quase_duplicatas(dados)

    if not quase_duplicatas:
        print("Nenhuma quase-duplicata encontrada.")
    else:
        print(f"{len(quase_duplicatas)} par(es) encontrado(s):")
        for mensagem_a, mensagem_b, similaridade in quase_duplicatas:
            print(f"  (sim={similaridade:.3f})")
            print(f"    A: {mensagem_a}")
            print(f"    B: {mensagem_b}")