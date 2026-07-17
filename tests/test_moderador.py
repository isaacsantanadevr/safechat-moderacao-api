import pytest

from classificador_semantico import MENSAGEM_BLOQUEADA
from moderador import censurar_mensagem


def test_preserva_mensagem_normal():
    mensagem = "A aula foi excelente"

    assert censurar_mensagem(mensagem) == mensagem


@pytest.mark.parametrize(
    "mensagem",
    [
        "td bem?",
        "eu te amo",
        "como você tá?",
        "perdi a cabeça",
        "eu te odeio",
        "te amo muito",
        "senti sua falta",
        "senti muita saudade",
        "o servidor caiu",
        "o aplicativo caiu hoje",
        "tenha cuidado na escada",
        "cuidado para não tropeçar",
        "isso não vai ficar pronto",
        "adoro conversar com você",
        "vai ficar assim mesmo",
        "pode deixar assim mesmo",
        "acho que vai ficar assim",
        "vai ficar desse jeito mesmo",
    ],
)
def test_preserva_conversas_cotidianas(mensagem):
    assert censurar_mensagem(mensagem) == mensagem


def test_censura_termo_exato_com_bag_of_words():
    mensagem = "Voce e um idiota"
    resultado = censurar_mensagem(mensagem)

    assert resultado != mensagem
    assert "idiota" not in resultado.lower()


def test_censura_variacao_com_jaccard():
    mensagem = "Voce e idiooota"
    resultado = censurar_mensagem(mensagem)

    assert resultado != mensagem
    assert "idiooota" not in resultado.lower()


def test_censura_ameaca_implicita_com_tfidf():
    assert censurar_mensagem("Eu sei onde voce mora") == MENSAGEM_BLOQUEADA


def test_censura_ameaca_contextual_com_tfidf():
    assert censurar_mensagem("Isso nao vai ficar assim") == MENSAGEM_BLOQUEADA
