from parameters.estimation.spme_benchmark_unimodal import (
    perform_spme_benchmark_unimodal
)
from parameters.estimation.spme_benchmark_multimodal import (
    perform_spme_benchmark_multimodal
)
perform_spme_benchmark_unimodal(verbose=True, seed=0)
for i in range(11):
    perform_spme_benchmark_multimodal(offset_indices=[i], verbose=True, seed=0)
perform_spme_benchmark_multimodal(
    offset_indices=[0, 2, 5, 10], verbose=True, seed=0
)
