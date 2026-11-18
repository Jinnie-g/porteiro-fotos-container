from porteiro.labels import derive_class, planificavel


def test_rule_section_2_4():
    base = {"qualidade": "Q1", "escopo": "face", "oclusao": "nenhuma", "obliquidade": "O1"}
    assert derive_class({**base, "n_castanhas": 4, "pontos_bem_espalhados": True}) == "P1"
    assert derive_class({**base, "n_castanhas": 4, "pontos_bem_espalhados": False, "corrugacao": "C0"}) == "P3"
    assert derive_class({**base, "corrugacao": "C2", "n_castanhas": 1}) == "P2"
    assert derive_class({**base, "corrugacao": "C2", "n_direcoes": 1}) == "P2"
    assert derive_class({**base, "corrugacao": "C1", "n_direcoes": 1}) == "P3"
    assert derive_class({**base, "obliquidade": "O3", "n_castanhas": 4, "pontos_bem_espalhados": True}) == "P4"
    assert derive_class({**base, "oclusao": ">50"}) == "P4"
    assert derive_class({**base, "qualidade": "Q3"}) == "P4"
    assert derive_class({"indeterminado": True}) == "INDETERMINADO"


def test_q3_points_are_not_localizable():
    a = {"qualidade": "Q3", "escopo": "face", "n_castanhas": 4, "pontos_bem_espalhados": True}
    assert derive_class(a) == "P4"


def test_binary_target():
    assert planificavel("P1") and planificavel("P2")
    assert not planificavel("P3") and not planificavel("P4")
    assert planificavel("INDETERMINADO") is None
