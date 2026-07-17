import pytest

from classificador_semantico import (
    CATEGORIA_SEGURA,
    LIMIAR_CONFIANCA,
    LIMIAR_SIMILARIDADE,
    _carregar_classificador,
    classificar_por_similaridade,
)


def test_classificador_reutiliza_cache_em_memoria():
    primeiro_carregamento = _carregar_classificador()
    segundo_carregamento = _carregar_classificador()

    assert primeiro_carregamento is segundo_carregamento


def test_classifica_ameaca_sem_palavra_proibida_literal():
    categoria, confianca, vizinhos = classificar_por_similaridade(
        "Eu sei onde voce mora"
    )

    assert categoria == "ameaca"
    assert confianca >= LIMIAR_CONFIANCA
    assert len(vizinhos) == 5


def test_texto_sem_relacao_com_dataset_e_considerado_seguro():
    categoria, confianca, vizinhos = classificar_por_similaridade("xyzqwv klmn")

    assert categoria == CATEGORIA_SEGURA
    assert confianca == 0.0
    assert vizinhos[0][2] < LIMIAR_SIMILARIDADE


def test_similaridade_de_cosseno_fica_entre_zero_e_um():
    _categoria, _confianca, vizinhos = classificar_por_similaridade(
        "A aula de hoje foi excelente"
    )

    assert all(0.0 <= similaridade <= 1.0 for _, _, similaridade in vizinhos)


def test_rejeita_quantidade_invalida_de_vizinhos():
    with pytest.raises(ValueError, match="k deve ser maior que zero"):
        classificar_por_similaridade("mensagem", k=0)
