

import matplotlib.pyplot as plt
from pathlib import Path

def main():    

    print("Hello, World")
    # make a figure
    fig, ax=plt.subplots()
    fig.suptitle("Hello, World")
    fig.savefig("figure.png")
    plt.close()
    return


if __name__ == "__main__":
    main()

