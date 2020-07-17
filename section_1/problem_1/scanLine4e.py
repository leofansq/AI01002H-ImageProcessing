"""
@leofansq
https://github.com/leofansq
"""
def scanLine4e(f, I, loc):
    """
    Parameters:
        f: Grayscale image
        I: An int num
        loc: 'row' or 'column'
    Return:
        A list of pixel value of the specific row/column
    """
    if loc == 'row':
        return f[I, :]
    elif loc == 'column':
        return f[:, I]
    else:
        raise ValueError("The third parameter should be row or column")