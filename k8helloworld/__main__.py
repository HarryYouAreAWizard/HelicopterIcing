

import matplotlib.pyplot as plt
from pathlib import Path

def main():    
    # make a figure
    fig, ax=plt.subplots()
    fig.savefig("figure.png")
    plt.close()
    return


if __name__ == "__main__":
    main()

