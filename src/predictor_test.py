from .predictor import apply


def test_algorithm():
    input = {"reaction": "[O-].C#CCC12CCC(=O)C=C1C(=C)CC1C3CCC(=O)C3(C)CCC12"}
    result = apply(input)
    return result