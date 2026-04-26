from numpy.typing import ArrayLike
from scipy.spatial.distance import jensenshannon

def calc_jsd(dis_1: ArrayLike, dis_2: ArrayLike):
    return jensenshannon(dis_1, dis_2, base=2)**2