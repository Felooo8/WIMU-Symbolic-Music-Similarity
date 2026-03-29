from numpy.typing import ArrayLike
from scipy.spatial.distance import jensenshannon

def calc_jsd(dataset_1: str, dataset_2: str, dis_1: ArrayLike, dis_2: ArrayLike):
    jsd_divergence = jensenshannon(dis_1, dis_2, base=2)**2
    print(f"\n[JSD] dataset_1={dataset_1} - dataset_2={dataset_2}: {jsd_divergence}")

    return jsd_divergence