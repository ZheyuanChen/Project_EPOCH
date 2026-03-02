import argparse
import os

import sdf_helper as sh


def read_sdf_file(file_path):
    """
    Reads an SDF file and returns the data object.
    """
    data = sh.getdata(file_path)
    return data



def examine_data_structure():
    """
    Examines the structure of the SDF data and lists available variables.
    """
    parser = argparse.ArgumentParser(description="EPOCH SDF to HDF5 Converter with Metadata")
    
    parser.add_argument(
        "--dir",
        dest="input",
        required=True, 
        help="Input directory containing SDF files"
    )
    args = parser.parse_args()
    args.input = os.path.abspath(args.input)


    data = sh.getdata(os.path.join(args.input, "0030.sdf"))
    sh.list_variables(data)
    return 1+1