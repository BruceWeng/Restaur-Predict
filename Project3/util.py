from math import sqrt

debug = True
def dprt(message, lv = 0):
    """
    Prints debug message

    Parameters
        message: The message to print
        lv: Indent message with this number of space
    """

    if debug:
        if lv == 2:
            message = "+ " + str(message)
        elif lv == 3:
            message = "> " + str(message)

        print "\033[01;32m[DEBUG] " + ("  " * lv) + message + "\033[00m"

def setlst(lst, index, value, func = None, newval = None):
    """
    Sets the index @ lst with value. Effectively extends lst by
    up to index if index is not found

    Parameters:
        lst: list
            The list to set

        index: int
            The index of lst to set

        value: any
            value to set to lst[index]

        func: function || None
            If None, simply sets lst[index]. Otherwise,
            lst[index] = func(lst[index], value)

        newval: any || None
            If lst[index] does not exist, init to newval

    Examples:
        a = []                                  # a = []
        setlst(a, 3, 5)                         # a = [None, None, None, 5]
        setlst(a, 5, 2)                         # a = [None, None, None, 5, None, 2]
        setlst(a, 5, 3)                         # a = [None, None, None, 5, None, 3]
        setlst(a, 3, 2, lambda x, y: x + y)     # a = [None, None, None, 7, None, 3]
    """

    lst.extend([newval] * (index - len(lst) + 1))

    if func != None and lst[index] != None:
        value = func(lst[index], value)

    lst[index] = value


def normalize(vec, smoothing = 0):
    """
    Normalize the given vector. This does not mutate the vector.

    Parameters:
        vec: listof(number)
            The vector to normalize

    Returns: listof(numbers)
        Normalezed vector (does not mutate the given vector)
    """
    # Smooth to avoid vector of 0
    smoothed = vec if smoothing == 0 else map(lambda x: x + smoothing, vec)
    # Calculate the magnitude
    magnitude = sqrt(sum(map(lambda x: x * x, smoothed)))
    # Normalize vector
    normalized = map(lambda x: x / magnitude, smoothed)

    return normalized


def stringify(lst, delim = ','):
    """ Converts a list to a separated list

    Parameters
    ----------
        lst: list(anything)
            A list of anything to join together

        delim: str
            Delimiter for joining the list. Default is comma (,)

    Returns
    -------
        A joint list in string form
    """

    return delim.join(str(item) for item in lst)

def listify(s, delim = ',', type = int):
    """ Converts a string into a list of the specified type

    Parameters
    ----------
        s: str
            The list to separate

        delim: str
            Delimiter for splitting the list. Default is comma (,)

        type: function
            Type conversion function to use for the resulting list. Default is int

    Returns
    -------
        A splitted string in list
    """

    return [type(x) for x in s.split(delim)]
