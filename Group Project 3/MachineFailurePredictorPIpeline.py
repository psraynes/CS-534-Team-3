from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

from data_processing import *


def main():
    data = load_and_sample('ai4i2020.csv', 'Machine failure')

    print(data.shape)
    

if __name__ == "__main__":
    main()
