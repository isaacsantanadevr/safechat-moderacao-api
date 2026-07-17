"""Camada de classificação textual com embeddings TF-IDF e k-NN.

As duas primeiras camadas do moderador tratam termos conhecidos: a lista de
palavras (Bag of Words) encontra correspondências exatas e o Jaccard encontra
variações de grafia. Esta terceira camada compara a mensagem inteira com os
exemplos rotulados em ``dados/brutos/mensagens.csv``.

O TF-IDF transforma textos em vetores esparsos sem exigir download de modelo
nem inferência neural. São combinados n-gramas de palavras, para reconhecer
expressões, e n-gramas de caracteres, para tolerar variações de escrita. O
vetorizador é ajustado no primeiro uso e permanece em memória.

O k-NN usa similaridade de cosseno. Cada vizinho vota com peso proporcional à
similaridade, e um limiar conservador impede decisões baseadas apenas em
palavras comuns de mensagens sem relação suficiente com o dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

CAMINHO_DATASET = Path(__file__).with_name("dados") / "brutos" / "mensagens.csv"

K_VIZINHOS = 5
LIMIAR_CONFIANCA = 0.55

# TF-IDF mede proximidade lexical, não compreensão profunda. Um limiar alto
# faz esta camada falhar de forma segura: só bloqueia textos realmente próximos
# dos exemplos ofensivos, em vez de adivinhar com base em palavras comuns.
LIMIAR_SIMILARIDADE_MINIMA = 0.60

# Alias mantido para compatibilidade com chamadas e testes anteriores.
LIMIAR_SIMILARIDADE = LIMIAR_SIMILARIDADE_MINIMA
CATEGORIA_SEGURA = "normal"

MENSAGEM_BLOQUEADA = (
    "[mensagem removida pela moderacao - conteudo sinalizado por analise semantica]"
)

_classificador_cache: dict | None = None


def _criar_vetorizador_tfidf() -> Pipeline:
    """Cria a mesma representação TF-IDF para produção e avaliação."""
    atributos = FeatureUnion(
        [
            (
                "palavras",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "caracteres",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("atributos", atributos),
            ("normalizar", Normalizer(copy=False)),
        ]
    )


def _selecionar_exemplos_sem_termo_explicito(dados: pd.DataFrame) -> pd.DataFrame:
    """Seleciona somente exemplos que realmente chegam à terceira camada.

    Mensagens com ``termos_ofensivos`` preenchido já são responsabilidade do
    Bag of Words e do Jaccard. Incluí-las faria o TF-IDF associar contextos
    neutros, como "perdi o ônibus", ao palavrão que aparece junto no exemplo.
    """
    sem_termo_explicito = (
        dados["termos_ofensivos"].fillna("").astype(str).str.strip().eq("")
    )
    return dados[sem_termo_explicito].reset_index(drop=True)


def _carregar_classificador() -> dict:
    """Ajusta o TF-IDF uma única vez e mantém os dados em memória."""
    global _classificador_cache

    if _classificador_cache is not None:
        return _classificador_cache

    dados = pd.read_csv(CAMINHO_DATASET)
    dados = _selecionar_exemplos_sem_termo_explicito(dados)
    mensagens = dados["mensagem"].astype(str).tolist()
    vetorizador = _criar_vetorizador_tfidf()
    embeddings = vetorizador.fit_transform(mensagens)

    _classificador_cache = {
        "vetorizador": vetorizador,
        "embeddings": embeddings,
        "categorias": dados["categoria"].to_numpy(),
        "mensagens": mensagens,
    }
    return _classificador_cache


def _calcular_similaridades(vetor, embeddings_referencia) -> np.ndarray:
    """Calcula cossenos aceitando matrizes esparsas ou arrays densos."""
    if hasattr(vetor, "toarray"):
        resultado = embeddings_referencia @ vetor.T
    else:
        resultado = embeddings_referencia @ np.asarray(vetor).reshape(-1)

    if hasattr(resultado, "toarray"):
        resultado = resultado.toarray()
    return np.asarray(resultado).ravel()


def _classificar_vetor(
    vetor,
    embeddings_referencia,
    categorias_referencia: np.ndarray,
    k: int,
    limiar_confianca: float,
    limiar_similaridade_minima: float,
) -> tuple[str, float, list[tuple[int, float]]]:
    """Executa o k-NN ponderado para produção e avaliação treino/teste."""
    if k < 1:
        raise ValueError("k deve ser maior que zero.")
    if len(categorias_referencia) == 0:
        return CATEGORIA_SEGURA, 0.0, []

    similaridades = _calcular_similaridades(vetor, embeddings_referencia)
    k_utilizado = min(k, len(categorias_referencia))
    indices_top_k = np.argsort(-similaridades)[:k_utilizado]
    vizinhos = [(int(i), float(similaridades[i])) for i in indices_top_k]

    maior_similaridade = vizinhos[0][1] if vizinhos else 0.0
    if maior_similaridade < limiar_similaridade_minima:
        return CATEGORIA_SEGURA, 0.0, vizinhos

    peso_por_categoria: dict[str, float] = {}
    for indice, similaridade in vizinhos:
        categoria = categorias_referencia[indice]
        peso_por_categoria[categoria] = (
            peso_por_categoria.get(categoria, 0.0) + max(similaridade, 0.0)
        )

    categoria_prevista = max(peso_por_categoria, key=peso_por_categoria.get)
    peso_total = sum(peso_por_categoria.values())
    confianca = (
        peso_por_categoria[categoria_prevista] / peso_total
        if peso_total > 0
        else 0.0
    )

    if confianca < limiar_confianca:
        return CATEGORIA_SEGURA, confianca, vizinhos
    return categoria_prevista, confianca, vizinhos


def classificar_por_similaridade(
    mensagem: str,
    k: int = K_VIZINHOS,
    limiar_confianca: float = LIMIAR_CONFIANCA,
    limiar_similaridade_minima: float = LIMIAR_SIMILARIDADE_MINIMA,
) -> tuple[str, float, list[tuple[str, str, float]]]:
    """Classifica uma mensagem pelos exemplos mais próximos do dataset.

    Retorna ``(categoria, confiança, vizinhos)``. A confiança é a parcela da
    similaridade total que sustenta a categoria vencedora. Cada vizinho contém
    ``(mensagem, categoria, similaridade)`` para depuração e explicação.
    """
    classificador = _carregar_classificador()
    vetor = classificador["vetorizador"].transform([mensagem])

    categoria, confianca, vizinhos_indices = _classificar_vetor(
        vetor,
        classificador["embeddings"],
        classificador["categorias"],
        k,
        limiar_confianca,
        limiar_similaridade_minima,
    )
    vizinhos = [
        (
            classificador["mensagens"][indice],
            classificador["categorias"][indice],
            similaridade,
        )
        for indice, similaridade in vizinhos_indices
    ]
    return categoria, confianca, vizinhos


def censurar_semantico(mensagem: str) -> str:
    """Bloqueia por inteiro uma mensagem ofensiva identificada pelo TF-IDF."""
    categoria, _confianca, _vizinhos = classificar_por_similaridade(mensagem)
    if categoria != CATEGORIA_SEGURA:
        return MENSAGEM_BLOQUEADA
    return mensagem


if __name__ == "__main__":
    mensagem_usuario = input("Digite uma mensagem: ")
    categoria, confianca, vizinhos = classificar_por_similaridade(mensagem_usuario)

    print(f"\nCategoria prevista: {categoria} (confiança: {confianca:.0%})")
    print("Vizinhos mais próximos no dataset:")
    for texto, categoria_vizinha, similaridade in vizinhos:
        print(f"  [{categoria_vizinha}] (sim={similaridade:.2f}) {texto}")
