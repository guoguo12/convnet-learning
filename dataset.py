import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

CIRCLES_PATH = 'full-numpy_bitmap-circle.npy'
SQUARES_PATH = 'full-numpy_bitmap-square.npy'


def load(egs_per_class=10000, one_hot=True):
    X_circles = np.load(CIRCLES_PATH)[:egs_per_class]
    X_squares = np.load(SQUARES_PATH)[:egs_per_class]
    X = np.vstack([X_circles, X_squares])

    y_circles = np.ones([egs_per_class, 1])
    y_squares = np.zeros([egs_per_class, 1])
    y = np.vstack([y_circles, y_squares])

    if one_hot:
        enc = OneHotEncoder(sparse=False)
        y = enc.fit_transform(y)

    return train_test_split(X, y, test_size=0.33, random_state=42)

if __name__ == '__main__':
    for mat in load(egs_per_class=10):
        print(mat)
