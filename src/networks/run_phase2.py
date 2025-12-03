import config as cfg
cfg.setup_paths()

from phase2 import pmi_pairings
from phase2 import build_networks       
from phase2 import graph_features
from phase2 import modeling_phase2
from phase2 import visualizations_phase2
import temporary_cleaning as temp_clean


def main():

    temp_clean.main()

    # 1. Compute PMI pairings
    pmi_pairings.main()

    # 2. Build ingredient network + centrality tables
    build_networks.main()

    # 3. Compute recipe-level graph features
    graph_features.main()

    # 4. Run phase 2 modeling (RF with SVD + graph features)
    modeling_phase2.main()

    # 5. Generate plots + network map
    visualizations_phase2.main()


if __name__ == "__main__":
    main()
