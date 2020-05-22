import argparse
from pathlib import Path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov']


def dir_path(path):
    """Wrapper around Path that ensures this directory is created."""
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def file_path(path):
    """Wrapper around Path that ensures parent directories are created.

        x = mkdir_parents(dir / video_with_dir_prefix)
    is short-hand for
        x = Path(dir / video_with_dir_prefix)
        x.parent.mkdir(exist_ok=True, parents=True)
    """
    if not isinstance(path, Path):
        path = Path(path)
    path.resolve().parent.mkdir(exist_ok=True, parents=True)
    return path


def glob_ext(path, extensions, recursive=False):
    if not isinstance(path, Path):
        path = Path(path)
    if recursive:
        # Handle one level of symlinks.
        path_children = list(path.glob('*'))
        all_files = list(path_children)
        for x in path_children:
            if x.is_dir():
                all_files += x.rglob('*')
    else:
        all_files = path.glob('*')
    return [
        x for x in all_files if any(x.name.endswith(y) for y in extensions)
    ]


def find_file_extensions(folder, stem, possible_extensions):
    if not isinstance(folder, Path):
        folder = Path(folder)
    for ext in possible_extensions:
        if ext[0] != '.':
            ext = f'.{ext}'
        path = folder / f'{stem}{ext}'
        if path.exists():
            return path
    return None


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def simple_table(rows):
    lengths = [
        max(len(row[i]) for row in rows) + 1 for i in range(len(rows[0]))
    ]
    row_format = ' '.join(('{:<%s}' % length) for length in lengths[:-1])
    row_format += ' {}'  # The last column can maintain its length.

    output = ''
    for i, row in enumerate(rows):
        if i > 0:
            output += '\n'
        output += row_format.format(*row)
    return output


def parse_bool(arg):
    """Parse string to boolean.
    Using type=bool in argparse does not do the right thing. E.g.
    '--bool_flag False' will parse as True. See
    <https://stackoverflow.com/q/15008758/1291812>

    Usage:
        parser.add_argument( '--choice', type=parse_bool)
    """
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected 'True' or 'False'.")
