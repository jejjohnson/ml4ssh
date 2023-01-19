from pathlib import Path
from typing import List


def check_if_directory(directory: str):

    if not Path(directory).exists():
        raise ValueError(f"Directory doesn't exist...: {directory}")
    else:
        return True


def check_if_file(directory: str):

    if not Path(directory).is_file():
        raise ValueError(f"File doesn't exist...: {directory}")
    else:
        return True


def list_all_files(
    directory: str, ext: str = "**/*", full_path: bool = True
) -> List[str]:

    # check if directory exists
    check_if_directory(directory)

    files = [x for x in Path(directory).glob(ext) if x.is_file()]

    if not full_path:
        files = get_file_names(files)

    return files


def get_file_names(files: List[str]):

    # convert to Path file
    files = list(map(lambda x: Path(x), files))

    # get file names
    files = list(map(lambda x: x.name, files))

    return files


def list_of_files_to_dict(files: List[str]):

    files = {Path(x).name: Path(x) for x in files}

    return files


def check_list_equal_elem(list1: List[str], list2: List[str]):

    list1.sort()
    list2.sort()

    if list1 == list2:
        return True
    else:
        return False


def get_subset_elements(sublist, suplist):

    combined_list = []
    for ilist1 in sublist:
        result = [x for x in suplist if ilist1 in str(x)]
        combined_list += result

    return combined_list


def get_subset_files_str(files_list: List[str], element: str):

    assert isinstance(element, str)

    files = list(filter(lambda ifile: element in str(ifile), files_list))

    return files


def get_subset_files_list(files_list: List[str], elements: List[str]):

    assert isinstance(elements, list)

    files = list()
    for ielement in elements:
        files += get_subset_files_str(files_list, ielement)

    return files
