import numpy as np
import pandas as pd


def regressao(arquivo):
    df = pd.read_csv(arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    # matriz X: coluna de 1s, area e quartos
    X = np.column_stack((np.ones(len(area)), area, quartos))
    Y = np.array(preco, dtype=float).reshape(-1, 1)

    # beta = (X^T X)^-1 X^T Y
    beta = np.linalg.pinv(X.T @ X) @ X.T @ Y

    return [float(beta[0][0]), float(beta[1][0]), float(beta[2][0])]


if __name__ == "__main__":
    b0, b1, b2 = regressao("imoveis_lar_perfeito.csv")
    print("Preco = %.2f + %.2f * Area + %.2f * Quartos" % (b0, b1, b2))
