"""
Utilitarios compartilhados para geracao e cache de embeddings de sentenca.

Usado pelo classificador semantico (classificador_semantico.py), pela
descoberta de novos termos (descobrir_termos.py) e pela deteccao de
quase-duplicatas (src/validar_dados.py) - qualquer rotina que precise
comparar mensagens por SIGNIFICADO, e nao so por caractere.

Modelo escolhido: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers).
E multilingue (cobre portugues), roda em CPU sem problema e tem download
pequeno (~470MB) comparado a modelos maiores. Na primeira execucao ele e
baixado automaticamente do Hugging Face Hub e fica em cache local do proprio
sentence-transformers (~/.cache/huggingface) - depois disso roda offline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

NOME_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
DIRETORIO_CACHE = Path(__file__).with_name("dados") / "cache"

_modelo = None  # singleton carregado sob demanda


def carregar_modelo():
    """Carrega o modelo de embeddings uma unica vez por processo.

    O import do sentence-transformers fica dentro da funcao (nao no topo do
    arquivo) de proposito: scripts que so usam a regex ou o Jaccard
    (moderador.py, similaridade.py) nao devem pagar o custo de carregar
    torch/transformers se a camada semantica nunca for chamada.
    """
    global _modelo

    if _modelo is None:
        from sentence_transformers import SentenceTransformer

        _modelo = SentenceTransformer(NOME_MODELO)

    return _modelo


def gerar_embeddings(textos: list[str]) -> np.ndarray:
    """Converte uma lista de mensagens em vetores de significado.

    normalize_embeddings=True deixa cada vetor com norma 1, o que permite
    usar produto escalar (mais rapido) no lugar de similaridade de cosseno
    completa - os dois resultados sao equivalentes quando os vetores sao
    normalizados.
    """
    modelo = carregar_modelo()
    vetores = modelo.encode(
        textos,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vetores, dtype=np.float32)


def _hash_conteudo(caminho: Path) -> str:
    """Hash do arquivo fonte (ex: mensagens.csv), usado para invalidar o
    cache automaticamente quando o dataset e alterado."""
    return hashlib.sha256(caminho.read_bytes()).hexdigest()[:16]


def _caminho_cache(nome_base: str, hash_dados: str) -> Path:
    DIRETORIO_CACHE.mkdir(parents=True, exist_ok=True)
    return DIRETORIO_CACHE / f"{nome_base}_{hash_dados}.npz"


def carregar_ou_gerar_embeddings(
    nome_base: str,
    caminho_fonte: Path,
    textos: list[str],
) -> np.ndarray:
    """Gera embeddings para `textos` ou reaproveita o cache em disco.

    Gerar embedding para ~600 mensagens e barato, mas nao ha motivo pra
    recalcular a cada chamada de API: o cache e valido enquanto o conteudo
    de `caminho_fonte` (o CSV) nao mudar. Se o CSV mudar, o hash muda, um
    novo arquivo de cache e criado e os embeddings sao recalculados.
    """
    hash_dados = _hash_conteudo(caminho_fonte)
    caminho = _caminho_cache(nome_base, hash_dados)

    if caminho.exists():
        return np.load(caminho)["embeddings"]

    embeddings = gerar_embeddings(textos)
    np.savez_compressed(caminho, embeddings=embeddings)
    return embeddings
