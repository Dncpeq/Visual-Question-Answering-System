
import os, re, torch, json
from os import walk
from os.path import join, splitext
from typing import Optional, Generator, List, Iterable, Dict

def get_paths(*paths: str,
              current: bool = False,
              static: bool = False,
              base_name: bool = False,
              file_only: bool = False,
              directory_only: bool = False,
              suffixes: Optional[List[str]] = None,
              skip_hidden: bool = True
              ) -> Generator[str, None, None]:
    """
    Yield file and/or directory paths under *root* that satisfy a set of
    filters.

    Args:
        paths: directories.
        current: If ``True``, limit the search to ``root`` itself (no recursion).
        static:  
            ``True`` take a snapshot of the whole tree first, then
              iterate it (newly-created files/dirs are **ignored**).  
            ``False`` stream results during traversal; anything
              created in a directory **before** the walker reaches it
              will be seen.
        base_name: Yield just the file/dir name instead of an absolute or relative path.
        file_only: Yield only files.
        directory_only: Yield only directories.
        suffixes: Iterable of allowed file extensions (case-insensitive,
            without the leading dot), or ``None`` to disable the filter.
        skip_hidden: Skip dot-files and dot-directories.

    Yields:
        str: Paths (or names if *base_name* is ``True``) that match the
        requested criteria.
    """

    def _filter_hidden(names: Iterable[str]) -> List[str]:
        """Remove dot-files / dot-dirs from *names*."""
        return [n for n in names if not n.startswith('.')]

    def _filter_suffixes(files: Iterable[str], suffixes: Optional[List[str]]) -> List[str]:
        """
        Keep only *files* whose extension (without the leading dot) is
        contained in *suffixes*.

        Args:
            files: File names to filter.
            suffixes: Iterable of allowed extensions (e.g. ``["png", "jpg"]``)
                or ``None`` to disable the filter.

        Returns:
            A list with the filtered file names.
        """
        if not suffixes:
            return list(files)
        exts = {s.lstrip('.').lower() for s in suffixes}
        return [f for f in files if splitext(f)[1][1:].lower() in exts]

    def _emit(rpath: str, dirs: List[str], files: List[str]):
        if not directory_only:
            for f in files:
                yield f if base_name else join(rpath, f)
        if not file_only:
            for d in dirs:
                yield d if base_name else join(rpath, d)

    root = join(*paths)

    # -------------------------------------------------
    # 1) Build a snapshot list first (if static)
    # -------------------------------------------------
    if static:
        snapshot: List[str] = []
        for r, dirs, files in walk(root):
            if current and r != root:
                continue

            if skip_hidden:
                dirs[:]  = _filter_hidden(dirs)
                files[:] = _filter_hidden(files)

            files = _filter_suffixes(files, suffixes)
            snapshot.extend(_emit(r, dirs, files))

        # Finished scanning — now nothing created afterwards can slip in
        for path in snapshot:
            yield path

    # -------------------------------------------------
    # 2) Stream as we go (dynamic)
    # -------------------------------------------------
    else:
        for r, dirs, files in walk(root):
            if current and r != root:
                continue

            if skip_hidden:
                dirs[:]  = _filter_hidden(dirs)
                files[:] = _filter_hidden(files)

            files = _filter_suffixes(files, suffixes)
            yield from _emit(r, dirs, files)


def find_all_occurrences(s: str, sub: str) -> List[int]:
    """
    Find all start indices of substring `sub` in string `s`.

    Args:
        s (str): The string to search within.
        sub (str): The substring to find.

    Returns:
        List[int]: A list of start indices where `sub` is found in `s`.
    """
    indices: List[int] = []
    start = 0

    while True:
        idx = s.find(sub, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + 1  # move past this match to find overlapping occurrences

    return indices


def find_all_occurrences(s: str, sub: str, overlap: bool = False) -> List[int]:

    """
    Find all start indices of substring `sub` in string `s`.
    
    Args:
        s (str): The string to search within.
        sub (str): The substring to find.
        overlap (bool): If True, allow overlapping matches. Default is False.
        
    Returns:
        List[int]: A list of start indices where `sub` occurs in `s`.
    """
    
    indices: List[int] = []
    if sub:
        start = 0
        step = 1 if overlap else len(sub)
        
        while True:
            idx = s.find(sub, start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + step
    
    return indices

def find_all_regex(s: str, sub: str, overlap: bool = False) -> List[int]:
    """
    Find all start indices of substring `sub` in string `s` using regex.

    Args:
        s: The string to search in.
        sub: The substring to find.
        overlap: If True, return overlapping matches; otherwise, non-overlapping.

    Returns:
        A list of start indices where `sub` occurs in `s`.
    """
    # Treat empty pattern as no match
    if not sub:
        return []  # Prevent matching "" at every position

    if overlap:
        # Positive lookahead to capture overlapping matches
        pattern = re.compile(f'(?={re.escape(sub)})')
    else:
        # Standard finditer returns non-overlapping matches
        pattern = re.compile(re.escape(sub))
    return [m.start() for m in pattern.finditer(s)]


def load_json(file, mode = "r") -> Dict:
    with open(file, mode, encoding="utf-8") as f:
        content = f.read()
        content = content if content else "{}"
        dic = json.loads(content)
    return dic

def save_json(file, dic, mode = "w", indent = 4) -> None:
    with open(file, mode, encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=indent)


def slice_tensor(
    x: torch.Tensor,
    dim: int,
    start: int | None = None,
    stop:   int | None = None,
    step:  int | None = 1
) -> torch.Tensor:
    # Build a list of slice() objects, all ‘:’, except along `dim`
    slc = [slice(None)] * x.ndim
    slc[dim] = slice(start, stop, step)
    return x[tuple(slc)]


def add_dim(tensor: torch.Tensor, dim: int, dim_num: int):
    shape = list(tensor.shape)
    shape.insert(dim, dim_num)
    return tensor.unsqueeze(dim).expand(*shape)


def del_dim(tensor: torch.Tensor, dim: int, index = 0):
    return slice_tensor(tensor, dim, index, index + 1).squeeze(dim)


def check_file(path: str, create=False) -> bool:
    is_exist = os.path.exists(path)

    if is_exist: return True
    else:
        if create is False: return False

    if not is_exist:
        path_dir = os.path.dirname(path)
        if path_dir and not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)

    with open(path, mode = "a", encoding="utf-8"):
        pass

    return True


def extract_before_dot(text):
    match = re.match(r'^[^.]+', text)
    if match:
        return match.group(0)
    return ""

def get_module_param_size(model):
    module_param_size = {}
    for name, param in model.named_parameters():
        module = extract_before_dot(name)
        module_param_size[module] = module_param_size.get(module, 0) + param.numel()
