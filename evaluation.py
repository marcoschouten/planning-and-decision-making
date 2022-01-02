from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot as plt

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {ord}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y



def main():
    print("Hello World!")
    
    x = np.linspace(-2, 1, 20)
    coeffs = [0, 0, +1, 1,]
    y_true = PolyCoefficients(x, coeffs)


    noise = np.random.normal(0, .1, x.shape)
    y_pred = y_true + 2*noise
    plt.plot(x, y_true)
    plt.plot(x, y_pred)
    plt.show()

    mse = np.round_(mean_squared_error(y_true, y_pred), decimals=2, out=None)
    print('mse {}'.format(mse))
  


if __name__ == "__main__":
    main()