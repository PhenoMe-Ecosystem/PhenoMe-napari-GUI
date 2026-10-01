"""Whether a path names the kind of thing a box on the loading tab asks for.

No Qt here and no notion of a box being required: these answer what a path is, and the panel says
which boxes may be left empty.
"""

from pathlib import Path


def check_directory_exists(path: Path) -> bool:
    """ Check if a directory exists.

    Args:
        path: The path to the directory.

    Returns:
        True if the directory exists, False otherwise - a file at that path included.
    """
    return path.exists() and path.is_dir()


def check_csv_file_exists(path: Path) -> bool:
    """ Check that a path names an existing .csv file.

    The suffix is read without its case, `metadata.CSV` being the same file to anyone reading it.

    Args:
        path: The path to the file.

    Returns:
        True if a .csv file exists there, False otherwise - a directory named like one included.
    """
    return path.is_file() and path.suffix.lower() == '.csv'
