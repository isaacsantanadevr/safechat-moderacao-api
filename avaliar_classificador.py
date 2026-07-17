"""
Avaliacao formal do classificador semantico (k-NN sobre TF-IDF).

Cuidado importante (vazamento de dados): o vetorizador TF-IDF e ajustado
(fit) SO no conjunto de treino. Se ajustassemos no dataset inteiro (como o
cache de producao faz, por design - la nao ha essa preocupacao, e uma
API rodando de verdade) e so depois separassemos treino/teste, o
vocabulario already teria "visto" as mensagens de teste, e a nota ficaria
artificialmente otimista. Aqui isso e evitado de proposito.

Gera:
  - Um relatorio no terminal (precisao, revocacao, f1 por categoria).
  - Uma imagem com a matriz de confusao (matriz_confusao.png), pronta
    pra apresentacao.

Como rodar:
    python avaliar_classificador.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera a imagem sem precisar de tela grafica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from classificador_semantico import (
    CATEGORIA_SEGURA,
    K_VIZINHOS,
    LIMIAR_CONFIANCA,
    LIMIAR_SIMILARIDADE_MINIMA,
    _classificar_vetor,
)

CAMINHO_DATASET = Path(__file__).with_name("dados") / "brutos" / "mensagens.csv"
CAMINHO_SAIDA_GRAFICO = Path(__file__).with_name("matriz_confusao.png")

PROPORCAO_TESTE = 0.2
SEMENTE_ALEATORIA = 42
CATEGORIAS_EM_ORDEM = ["normal", "palavrao", "insulto", "ameaca"]


def _normalizar(matriz_esparsa) -> np.ndarray:
    """Mesma normalizacao usada em embeddings_utils.py (norma 1 por vetor,
    pra comparar por produto escalar em vez de cosseno completo)."""
    matriz_densa = np.asarray(matriz_esparsa.todense(), dtype=np.float32)
    normas = np.linalg.norm(matriz_densa, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz_densa / normas


def vetorizar_tfidf(textos_treino: list[str], textos_teste: list[str]):
    """Ajusta um TF-IDF novo, so no treino, e usa o mesmo vocabulario pra
    transformar o teste. Mesma configuracao do embeddings_utils.py atual -
    trocar essa funcao por uma versao Word2Vec/GloVe no futuro e o unico
    passo necessario pra reaproveitar todo o resto deste script."""
    vetorizador = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1
    )
    vetores_treino = _normalizar(vetorizador.fit_transform(textos_treino))
    vetores_teste = _normalizar(vetorizador.transform(textos_teste))
    return vetores_treino, vetores_teste


def avaliar(nome_metodo: str, funcao_vetorizacao) -> None:
    df = pd.read_csv(CAMINHO_DATASET)

    treino, teste = train_test_split(
        df,
        test_size=PROPORCAO_TESTE,
        stratify=df["categoria"],
        random_state=SEMENTE_ALEATORIA,
    )

    vetores_treino, vetores_teste = funcao_vetorizacao(
        treino["mensagem"].tolist(), teste["mensagem"].tolist()
    )
    categorias_treino = treino["categoria"].to_numpy()
    categorias_reais = teste["categoria"].to_numpy()

    previsoes = []
    for vetor in vetores_teste:
        categoria_prevista, _confianca, _vizinhos = _classificar_vetor(
            vetor,
            vetores_treino,
            categorias_treino,
            K_VIZINHOS,
            LIMIAR_CONFIANCA,
            LIMIAR_SIMILARIDADE_MINIMA,
        )
        previsoes.append(categoria_prevista)

    print(f"\n{'=' * 60}")
    print(f"Metodo: {nome_metodo}")
    print(f"Treino: {len(treino)} mensagens | Teste: {len(teste)} mensagens")
    print(f"{'=' * 60}\n")

    print(classification_report(
        categorias_reais, previsoes, labels=CATEGORIAS_EM_ORDEM, digits=2, zero_division=0
    ))

    matriz = confusion_matrix(categorias_reais, previsoes, labels=CATEGORIAS_EM_ORDEM)

    fig, eixo = plt.subplots(figsize=(6, 5))
    exibicao = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=CATEGORIAS_EM_ORDEM)
    exibicao.plot(ax=eixo, cmap="Purples", colorbar=False, values_format="d")
    eixo.set_title(f"Matriz de confusao - {nome_metodo}")
    fig.tight_layout()
    fig.savefig(CAMINHO_SAIDA_GRAFICO, dpi=150)
    plt.close(fig)

    print(f"Grafico salvo em: {CAMINHO_SAIDA_GRAFICO}")


if __name__ == "__main__":
    avaliar("TF-IDF (bag-of-words ponderado)", vetorizar_tfidf)