"""Code for computing "rolling checksums", and using it to compute how "new"
a piece of text is, compared to previously seen text.

A rolling checksum is simply a checksum that is computed over a sliding
N-byte substring of the data, and only takes O(1) time to update (as opposed
to naive solution of recomputing the checksum for that substring in O(N) time)

This code hasn't been thoroughly tested.  Use it as a starting point.
"""

def checksum_iter(data, window_size=20):
    """Usage:
    for checksum in checksum_iter("data...", 20):
        print "checksum for 20-byte window is:", checksum

    For a given 20-byte substring, we compute 2 state variables:
        a = sum_i(data[i])
        b = sum_i(data[i] * (window_size-i))
    And the checksum for that 20-byte substring is a simple function of a & b.
    The form of a & b are such that we can easily update them in constant time
    as we "slide" the window to the right.
    """
    assert type(data) == str
    len_data = len(data)

    # compute the state (& checksum) for the first 20-byte window.
    a = 0
    b = 0
    for i in xrange(window_size):
        a += ord(data[i])
        b += (window_size - i) * ord(data[i])

    yield (b << 16) | a  # the checksum

    # now update the state as we slide the window across.
    for i in xrange(window_size, len_data):
        byte_leaving = ord(data[i-window_size])
        byte_entering = ord(data[i])

        a -= byte_leaving
        b -= byte_leaving * window_size

        a += byte_entering
        b += a

        yield (b << 16) | a


def new_ratio(new_text, checksums, update_checksums=True):
    """new_text is a string.  checksums is a set of integers.
    Returns a value in [0,1] indicating what portion of new_text is 'new' with respect
    to checksums.  Also updates checksums as it goes along."""
    if type(new_text) == unicode:
        new_text = new_text.encode('utf-8')

    num_checksums = 0  # how many we extract from @new_text
    num_misses = 0  # how many of those were not in @checksums
    new_checksums = set()

    WINDOW_SIZE = 20
    for checksum in checksum_iter(new_text, WINDOW_SIZE):
        num_checksums += 1
        new_checksums.add(checksum)
        if checksum not in checksums:
            num_misses += 1

    if update_checksums:
        checksums.update(new_checksums)
    # see how many checksums have been added to @checksums
    return num_misses / float(num_checksums)

if __name__ == "__main__":
    checksums = set()
    all_text = set()  # set of strings we've processed
    for new_text in [
        "The rain in Spain falls mainly in the plain.",
        "The rain in Spain falls mainly in the plain.",
        "The rain in Spain falls mainly in the plane..",
        "The rain in Spain falls mainly in the plane.......",
        "The rain in Spain falls mainly in the plane................",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]:
        print "Based on the %d strings I've seen so far, '%s' is %d%% new" % (
            len(all_text), new_text, 100*new_ratio(new_text, checksums))
        all_text.add(new_text)
