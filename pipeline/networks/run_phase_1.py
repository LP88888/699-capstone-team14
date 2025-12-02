import config as cfg
cfg.setup_paths()

from phase1 import modeling_extended
from phase1 import visualization
import temporary_cleaning as temp_clean

def main():
 
    temp_clean.main()
    
    modeling_extended.main()

    visualization.main()


if __name__ == "__main__":  
    main()
