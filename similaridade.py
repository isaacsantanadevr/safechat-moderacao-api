"""
Camada de deteccao por similaridade de Jaccard.

Complementa a correspondencia exata (regex) do moderador.py: pega variacoes
de grafia que nao batem palavra-por-palavra com a lista de termos proibidos
(leetspeak, letras repetidas, espacamento entre letras, etc.), sem precisar
cadastrar manualmente cada variacao possivel.

Esta camada roda DEPOIS da regex, apenas sobre as palavras que sobraram sem
ser censuradas. Nao substitui a correspondencia exata, so complementa.
"""

import re

N_GRAMA = 2
LIMIAR_SIMILARIDADE = 0.6
LIMIAR_DISTANCIA_NORMALIZADA = 0.2
TAMANHO_MINIMO_TOKEN = 3

TABELA_NORMALIZACAO = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "$": "s", "@": "a", "(": "c",
})

PALAVRAS_PERMITIDAS = {
    "sabado", "bota",
}


def limpar_para_comparacao(token: str) -> str:
    """Desfaz truques comuns de disfarce ANTES de comparar (nao altera o
    texto original exibido pro usuario, so a copia usada internamente pra
    decidir se e suspeito):
    - leetspeak (0->o, 4->a, 3->e, 1->i, $->s, @->a, (->c)
    - separadores usados no meio da palavra (underscore, asterisco, ponto)
    """
    token = token.lower().translate(TABELA_NORMALIZACAO)
    token = re.sub(r"[_*.]", "", token)
    return token


def gerar_ngramas(texto: str, n: int = N_GRAMA) -> set[str]:
    if len(texto) < n:
        return {texto}
    return {texto[i:i + n] for i in range(len(texto) - n + 1)}


def similaridade_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersecao = len(a & b)
    uniao = len(a | b)
    return intersecao / uniao if uniao else 0.0


def distancia_edicao(a: str, b: str) -> int:
    """Distancia de Levenshtein: numero minimo de insercoes, remocoes ou
    substituicoes pra transformar 'a' em 'b'. Pega trocas pontuais de
    caractere (ex: f0der -> foder) que o Jaccard de bigramas subestima
    em palavras curtas."""
    m, n = len(a), len(b)
    linha_anterior = list(range(n + 1))

    for i in range(1, m + 1):
        linha_atual = [i] + [0] * n
        for j in range(1, n + 1):
            custo = 0 if a[i - 1] == b[j - 1] else 1
            linha_atual[j] = min(
                linha_anterior[j] + 1,
                linha_atual[j - 1] + 1,
                linha_anterior[j - 1] + custo,
            )
        linha_anterior = linha_atual

    return linha_anterior[n]


def distancia_normalizada(a: str, b: str) -> float:
    dist = distancia_edicao(a, b)
    maior = max(len(a), len(b))
    return dist / maior if maior else 0.0


def termos_de_uma_palavra(termos: list[str]) -> list[str]:
    """Jaccard por token so faz sentido comparando com termos de uma palavra
    so. Termos com espaco ou hifen (ex: 'puta que pariu', 'sem-vergonha')
    continuam cobertos pela correspondencia exata (regex)."""
    return [t for t in termos if " " not in t and "-" not in t]


def token_e_suspeito(
    token: str,
    termos_simples: list[str],
    limiar_jaccard: float = LIMIAR_SIMILARIDADE,
    limiar_distancia: float = LIMIAR_DISTANCIA_NORMALIZADA,
) -> bool:
    limpo = limpar_para_comparacao(token)

    if limpo in PALAVRAS_PERMITIDAS:
        return False

    if len(limpo) < TAMANHO_MINIMO_TOKEN:
        return False

    grama_token = gerar_ngramas(limpo)

    for termo in termos_simples:
        if similaridade_jaccard(grama_token, gerar_ngramas(termo)) >= limiar_jaccard:
            return True
        if distancia_normalizada(limpo, termo) <= limiar_distancia:
            return True

    return False


_PADRAO_TOKEN = re.compile(r"\w+", flags=re.UNICODE)
_PADRAO_SOLETRADO = re.compile(r"\w(?:[.\-*]\w){3,}", flags=re.UNICODE)


def mascarar_trecho(trecho: str) -> str:
    return "".join("*" if c.isalnum() else c for c in trecho)


def censurar_variacoes(mensagem: str, termos_proibidos: list[str]) -> str:
    """Camada extra que roda depois da correspondencia exata (regex).
    Cobre dois padroes que a regex sozinha nao pega:
      1) palavra soletrada com separador entre as letras (c-a-r-a-l-h-o)
      2) palavra com leetspeak/disfarce, comparada por Jaccard + distancia
         de edicao contra a lista de termos de uma palavra so

    Importante: a normalizacao (leetspeak) e usada so pra DECIDIR se uma
    palavra e suspeita e pra reunir palavras quebradas por simbolos como
    @ ou $ (que nao contam como "letra" pro Python). O texto original só
    é alterado nos trechos que realmente forem sinalizados como ofensivos
    - tudo o resto sai exatamente como entrou.
    """
    termos_simples = termos_de_uma_palavra(termos_proibidos)

    def substituir_soletrado(m: re.Match) -> str:
        trecho = m.group()
        candidato = re.sub(r"[.\-*]", "", trecho)
        if token_e_suspeito(candidato, termos_simples):
            return mascarar_trecho(trecho)
        return trecho

    mensagem = _PADRAO_SOLETRADO.sub(substituir_soletrado, mensagem)

    normalizado = mensagem.lower().translate(TABELA_NORMALIZACAO)
    pedacos = []
    posicao = 0

    for m in _PADRAO_TOKEN.finditer(normalizado):
        inicio, fim = m.span()
        pedacos.append(mensagem[posicao:inicio])

        trecho_original = mensagem[inicio:fim]
        if token_e_suspeito(m.group(), termos_simples):
            pedacos.append(mascarar_trecho(trecho_original))
        else:
            pedacos.append(trecho_original)

        posicao = fim

    pedacos.append(mensagem[posicao:])
    return "".join(pedacos)