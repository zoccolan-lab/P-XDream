from comparison_plots import main as comparison_plots_main
from metrics import main as metrics_main
from embeddings import main as embeddings_main
from labelings import main as labelings_main
import loguru


if __name__ == '__main__':
    
    labelings_main()
    comparison_plots_main()
    print('here')
    metrics_main()
    print('here')
    embeddings_main()