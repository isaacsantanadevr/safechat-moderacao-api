from pathlib import Path

import pandas as pd


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


if __name__ == "__main__":
    dados = carregar_dados()
    validar_dados(dados)