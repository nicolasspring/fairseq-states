import os

def exp_path_search(pattern):
    """
    Exponential search to find the next free path in a sequentially named list of files.
    Adapted from https://stackoverflow.com/a/47087513/ (Author: James https://stackoverflow.com/users/165783/james).

    Args:
        pattern (str): a string pattern pointing to a path (e.g. "./out-%s.pt")
    """
    i = 1
    while os.path.exists(pattern % i):
        i = i * 2

    interval_start, interval_end = (i // 2, i)
    while interval_start + 1 < interval_end:
        mid_point = (interval_start + interval_end) // 2
        interval_start, interval_end = (mid_point, interval_end) if os.path.exists(pattern % mid_point) \
                                       else (interval_start, mid_point)
    next_free = interval_end

    return pattern % next_free
