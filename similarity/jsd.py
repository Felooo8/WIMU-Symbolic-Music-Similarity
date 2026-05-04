from numpy.typing import ArrayLike
from scipy.spatial.distance import jensenshannon


def calc_jsd(*args: ArrayLike):
    if len(args) == 2:
        dis_1, dis_2 = args
    elif len(args) == 4:
        _, _, dis_1, dis_2 = args
    else:
        raise TypeError("calc_jsd expects 2 distributions or 2 labels plus 2 distributions")

    return jensenshannon(dis_1, dis_2, base=2) ** 2
