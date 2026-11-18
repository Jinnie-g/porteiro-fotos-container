"""Seção 2.4: classe de triagem derivada por regra a partir dos atributos rotulados.

O anotador registra atributos observáveis; a classe é calculada aqui, para
que a regra possa ser ajustada sem rotular tudo de novo.
"""
from __future__ import annotations

ATRIBUTOS = {
    "zoom_status": ["Z0", "Z1", "Z2"],
    "qualidade": ["Q1", "Q2", "Q3"],
    "oclusao": ["nenhuma", "<20", "20-50", ">50"],
    "escopo": ["face", "outro"],
    "corrugacao": ["C0", "C1", "C2"],
    "obliquidade": ["O1", "O2", "O3"],
    # inteiros: n_castanhas, n_vertices, n_direcoes; booleano: pontos_bem_espalhados
}


def n_pontos(a: dict) -> int:
    """Só conta pontos localizáveis (qualidade != Q3 e obliquidade != O3)."""
    if a.get("qualidade") == "Q3" or a.get("obliquidade") == "O3":
        return 0
    return int(a.get("n_castanhas") or 0) + int(a.get("n_vertices") or 0)


def derive_class(a: dict) -> str:
    if a.get("indeterminado"):
        return "INDETERMINADO"
    if (a.get("qualidade") == "Q3" or a.get("escopo") == "outro"
            or a.get("oclusao") == ">50" or a.get("obliquidade") == "O3"):
        return "P4"
    npts = n_pontos(a)
    if npts >= 4 and a.get("pontos_bem_espalhados", False):
        return "P1"
    if a.get("corrugacao") == "C2" and (int(a.get("n_castanhas") or 0) >= 1
                                         or int(a.get("n_direcoes") or 0) >= 1
                                         or 2 <= npts <= 3):
        return "P2"
    if a.get("qualidade") in ("Q1", "Q2"):
        return "P3"
    return "P4"


def planificavel(cls: str) -> bool | None:
    """Alvo binário principal: P1 ∪ P2."""
    if cls == "INDETERMINADO":
        return None
    return cls in ("P1", "P2")
