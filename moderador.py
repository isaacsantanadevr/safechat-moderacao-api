from pathlib import Path
import re

from similaridade import censurar_variacoes

ARQUIVO_TERMOS = (Path(__file__)).with_name("palavras_proibidas.txt")

def cargar_termos_proibidos() -> list[str]:

    with open(ARQUIVO_TERMOS, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]

def criar_padrao(termos: list[str]) -> re.Pattern:
    termos_ordenados = sorted(termos, key=len, reverse=True)
    termos_regex = [re.escape(termo) for termo in termos_ordenados]

    return re.compile(
        r"(?<!\w)(?:" + "|".join(termos_regex) + r")(?!\w)",
        flags=re.IGNORECASE
    )

def mascarar_trecho(trecho: str) -> str:
    return "".join(
        "*"if caractere.isalnum() else caractere for caractere in trecho
    )

termos_proibidos = cargar_termos_proibidos()
padrao_proibido = criar_padrao(termos_proibidos)

def censurar_mensagem(mensagem: str) -> str:
    mensagem_censurada = padrao_proibido.sub(
        lambda resultado: mascarar_trecho(resultado.group()), mensagem
    )

    try:
        mensagem_censurada = censurar_variacoes(mensagem_censurada, termos_proibidos)
    except Exception:
        pass

    return mensagem_censurada

if __name__ == "__main__":
    mensagem = input("Digite uma mensagem: ")
    mensagem_censurada = censurar_mensagem(mensagem)
    print("Mensagem censurada:", mensagem_censurada)