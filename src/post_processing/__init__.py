import sdf_helper as sh


def read_sdf_file(file_path):
    """
    Reads an SDF file and returns the data object.
    """
    data = sh.getdata(file_path)
    return data

def examine_data_structure(file_path):
    """
    Examines the structure of the SDF data and lists available variables.
    """
    data = sh.getdata(file_path)
    sh.list_variables(data)
    return data