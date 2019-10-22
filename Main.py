import pyswarms as ps
import Orange
import GEOparse
import Utils as ut
from Orange.data.pandas_compat import table_from_frame
from Bio import Geo

def main():
    dataset = GEOparse.get_GEO(geo="GDS360", destdir="./")
    print(dataset.table)
    data = Orange.data.Table("./datasetExplore")
    print(data.X[0][9470:])

if __name__== "__main__":
    main()
