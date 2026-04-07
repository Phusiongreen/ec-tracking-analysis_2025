from pathlib import Path

from nbconvert import PythonExporter

def nb_to_py(path_nb, filename, outpath_py, prefix_str="tmp_"):
    outpath_py = Path(outpath_py)

    # export
    exp = PythonExporter()
    nb_python_mem, _ = exp.from_filename(str(path_nb))

    p = outpath_py.joinpath(prefix_str + filename + ".py")
    with open(p, "w+") as f:
        f.write(nb_python_mem)

    return p