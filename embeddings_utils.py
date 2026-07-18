"""
Utilitarios compartilhados para geracao e cache de vetores de significado.

Gera vetores TF-IDF (scikit-learn) a partir de texto: nao baixa modelo,
nao depende de torch, so usa uma biblioteca que ja e dependencia do
projeto.

Usado pelo classificador semantico (classificador_semantico.py) e por
qualquer rotina futura que precise comparar mensagens por significado
aproximado, e nao so por caractere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DIRETORIO_CACHE = Path(__file__).with_name("dados") / "cache"

# Vetorizador ajustado (fit) uma unica vez, no dataset rotulado. Sem ele,
# nao da pra vetorizar uma mensagem nova: o TF-IDF depende do vocabulario
# aprendido no ajuste inicial pra saber quais palavras existem e o quao
# raras elas sao.
_vetorizador: TfidfVectorizer | None = None


def _criar_vetorizador() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),  # unigramas + bigramas: pega tambem expressoes curtas
        min_df=1,
    )


def _normalizar(matriz_esparsa) -> np.ndarray:
    """Deixa cada vetor com norma 1 (mesmo papel do normalize_embeddings de
    antes), pra comparar por produto escalar em vez de cosseno completo."""
    matriz_densa = np.asarray(matriz_esparsa.todense(), dtype=np.float32)
    normas = np.linalg.norm(matriz_densa, axis=1, keepdims=True)
    normas[normas == 0] = 1.0  # evita divisao por zero (mensagem vazia/so ruido)
    return matriz_densa / normas


def gerar_embeddings(textos: list[str]) -> np.ndarray:
    """Converte mensagens em vetores TF-IDF, usando o vocabulario ja
    ajustado no dataset rotulado.

    Precisa que `carregar_ou_gerar_embeddings` tenha rodado antes pelo
    menos uma vez no processo - e o que ajusta (fit) o vetorizador. O
    classificador semantico sempre faz isso primeiro (carrega o dataset
    antes de classificar qualquer mensagem nova).
    """
    if _vetorizador is None:
        raise RuntimeError(
            "Vetorizador TF-IDF ainda nao foi ajustado. Chame "
            "carregar_ou_gerar_embeddings() com o dataset rotulado antes."
        )

    vetores = _vetorizador.transform(textos)
    return _normalizar(vetores)


def _hash_conteudo(caminho: Path) -> str:
    """Hash do arquivo fonte (ex: mensagens.csv), usado para invalidar o
    cache automaticamente quando o dataset e alterado."""
    return hashlib.sha256(caminho.read_bytes()).hexdigest()[:16]


def _caminho_cache(nome_base: str, hash_dados: str) -> Path:
    DIRETORIO_CACHE.mkdir(parents=True, exist_ok=True)
    return DIRETORIO_CACHE / f"{nome_base}_{hash_dados}.joblib"


def carregar_ou_gerar_embeddings(
    nome_base: str,
    caminho_fonte: Path,
    textos: list[str],
) -> np.ndarray:
    """Ajusta (fit) o vetorizador TF-IDF no dataset e gera os vetores, ou
    reaproveita o cache em disco se o conteudo de `caminho_fonte` nao
    mudou desde a ultima vez (mesma logica de antes, so troca o formato
    salvo - agora e o vetorizador inteiro, via joblib, nao so os numeros).
    """
    global _vetorizador

    hash_dados = _hash_conteudo(caminho_fonte)
    caminho = _caminho_cache(nome_base, hash_dados)

    if caminho.exists():
        _vetorizador, vetores = joblib.load(caminho)
        return vetores

    _vetorizador = _criar_vetorizador()
    matriz_esparsa = _vetorizador.fit_transform(textos)
    vetores = _normalizar(matriz_esparsa)

    joblib.dump((_vetorizador, vetores), caminho)
    return vetores