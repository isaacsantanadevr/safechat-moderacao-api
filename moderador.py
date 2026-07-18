from pathlib import Path
import re

from similaridade import censurar_variacoes, mascarar_trecho
from classificador_semantico import censurar_semantico

ARQUIVO_TERMOS = (Path(__file__)).with_name("palavras_proibidas.txt")

def carregar_termos_proibidos() -> list[str]:
    """Lê a lista usada pela camada de correspondência exata (Bag of Words)."""

    with open(ARQUIVO_TERMOS, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]

def criar_padrao(termos: list[str]) -> re.Pattern:
    """Monta um regex único com todos os termos proibidos (Bag of Words).

    Ordena do termo mais longo pro mais curto antes de juntar com "|":
    isso evita que um termo curto "roube" o match de um termo maior que
    o contém`(?<!\w)...(?!\w)` garante que só bate palavra inteira, não pedaço
    dentro de outra palavra.
    """
    termos_ordenados = sorted(termos, key=len, reverse=True)
    termos_regex = [re.escape(termo) for termo in termos_ordenados]

    return re.compile(
        r"(?<!\w)(?:" + "|".join(termos_regex) + r")(?!\w)",
        flags=re.IGNORECASE
    )

# Carregado uma vez só, no import do módulo - a lista e o regex ficam
# prontos em memória pra toda chamada de censurar_mensagem.
termos_proibidos = carregar_termos_proibidos()
padrao_proibido = criar_padrao(termos_proibidos)

def censurar_mensagem(mensagem: str) -> str:
    """Aplica as três camadas de moderação em ordem crescente de custo.

    1) Regex (Bag of Words) - correspondência exata, rápida.
    2) Jaccard/Levenshtein (similaridade.py) - pega disfarces de grafia.
    3) TF-IDF/k-NN (classificador_semantico.py) - só roda se as duas
       primeiras não acharam nada, porque é a mais cara e o objetivo é
       gastar esse custo só quando realmente precisa.
    """
    mensagem_censurada = padrao_proibido.sub(
        lambda resultado: mascarar_trecho(resultado.group()), mensagem
    )

    # se a camada de similaridade falhar por
    # qualquer motivo, a API não deve cair - ela só segue com o texto que já
    # tinha até então, em vez de devolver erro 500 pro SafeChat.
    try:
        mensagem_censurada = censurar_variacoes(mensagem_censurada, termos_proibidos)
    except Exception:
        pass

    # A camada 3 usa embeddings TF-IDF somente quando as camadas 1 e 2 não
    # sinalizam nada. O ajuste ocorre uma vez e as próximas consultas usam o
    # vetorizador e o dataset já mantidos em memória.
    if mensagem_censurada == mensagem:
        try:
            mensagem_censurada = censurar_semantico(mensagem_censurada)
        except Exception:
            pass

    return mensagem_censurada

if __name__ == "__main__":
    mensagem = input("Digite uma mensagem: ")
    mensagem_censurada = censurar_mensagem(mensagem)
    print("Mensagem censurada:", mensagem_censurada)
