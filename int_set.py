"""
Example usage:

    int_set = IntSet(max_int=20)
    assert 10 not in int_set
    int_set.add(10)
    assert 10 in int_set

An IntSet stores a set of integers in the range [0, max_int].
It supports the basic set() operations: add(), remove(), len(), etc...
It uses a bit-array to represent the set, so the memory usage is about max_int/8 bytes.
"""

class IntSet:
    def __init__(self, max_int):
        self._bytes = bytearray("\0" * (max_int/8 + 1))
        self._num_ints = 0

    def __contains__(self, x):
        """Note: if x is outside the valid range, returns false."""
        if not (0 <= x/8 < len(self._bytes)):
            return False
        return bool(self._bytes[x/8] & (1 << (x%8)))

    def __len__(self):
        return self._num_ints

    def add(self, x):
        """Note: if x is outside the valid range, will raise an IndexError"""
        byte = self._bytes[x/8]
        new_byte = byte | (1 << (x%8))

        if byte == new_byte: return  # member already in set

        self._bytes[x/8] = new_byte
        self._num_ints += 1

    def remove(self, x):
        """Note: if x is outside the valid range, will raise an IndexError"""
        byte = self._bytes[x/8]
        new_byte = byte & (255 ^ (1 << (x%8)))

        if byte == new_byte: return  # member already not in set

        self._bytes[x/8] = new_byte
        self._num_ints -= 1

    def __iter__(self):
        for byte_i in xrange(len(self._bytes)):
            byte = self._bytes[byte_i]
            # iterate through all bits, starting with least-significant bit
            for bit_j in xrange(8):
                if byte & (1 << bit_j):
                    yield (byte_i*8) + bit_j

def test_IntSet_basic():
    int_set = IntSet(4)
    int_set.add(2)
    int_set.add(4)
    assert [2,4] == [x for x in int_set]

def test_IntSet_advanced():
    int_set = IntSet(17)

    # can't add far outside of range
    try:
        int_set.add(100)
    except:
        pass
    else:
        assert False

    # but it's okay to test membership outside the range
    assert 18 not in int_set
    assert -1 not in int_set

    # let's add (and then remove) each member, one by one
    for i in xrange(0, 17):
        # at first, it's not in the set
        assert i not in int_set

        # now check after you've added it
        int_set.add(i)
        assert i in int_set

        # adding a second time is okay
        int_set.add(i)
        assert i in int_set

        # removing from the set should work
        int_set.remove(i)
        assert i not in int_set

        # removing a second time is okay
        int_set.remove(i)
        assert i not in int_set

    # should be nothing left after all that.
    assert len(int_set) == 0


if __name__ == "__main__":
    test_IntSet_basic()
    test_IntSet_advanced()
