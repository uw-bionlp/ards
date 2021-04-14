



import difflib
from collections import Counter, OrderedDict
import re




def matcher_str(label, i1, i2, j1, j2, x, y):
    return '{:7}   x[{}:{}] --> y[{}:{}] {!r:>8} --> {!r}'.format( \
                    label, i1, i2, j1, j2, x[i1:i2], y[j1:j2])



def align_whitespace(x, y):
    """
    align strings to create a character map from x to y

    """

    # , Double check that non - whitespace characters match
    x_chars = ''.join(x.split())
    y_chars = ''.join(y.split())
    assert x_chars == y_chars, '''"{}" vs "{}"'''.format(x_chars, y_chars)

    # output variable, character map from x to y
    char_map = OrderedDict()

    # create sequence matcher object where all white is junk
    # s = difflib.SequenceMatcher(None, x, y, autojunk=False)
    func = lambda x: x in [" ", '\t', '\n']
    s = difflib.SequenceMatcher(func, x, y, autojunk=False)

    # iterator over aligned portions of strings
    for label, i1, i2, j1, j2 in s.get_opcodes():

        I = list(range(i1, i2))
        J = list(range(j1, j2))

        x_tmp = x[i1:i2]
        y_tmp = y[j1:j2]

        #print(matcher_str(label, i1, i2, j1, j2, x, y))

        if label == "equal":
            assert len(I) == len(J)
            for i, j in zip(I, J):
                char_map[i] = j

        elif label == "delete":
            assert j1 == j2
            assert x_tmp.isspace(), f'''\n{repr(x_tmp)} from\n {repr(x)} \nnot in\n{repr(y)}'''
            for i in I:
                char_map[i] = j1

        elif label == "insert":
            assert i1 == i2
            assert y_tmp.isspace()
            char_map[i1] = J[0]

        elif label == "replace":
            assert x_tmp.isspace()
            assert y_tmp.isspace()

            for i, j in zip(I, J):
                char_map[i] = j
        else:
            raise ValueError

    char_map[i2] = j2

    # make sure all characters have mapping
    #assert len(char_map) == len(x)
    for i in range(len(x)+1):
        assert i in char_map

    return char_map



def get_matching_text(source, target):
    '''

    '''
    # Get non space non blank lines
    lines = [line for line in source.splitlines() if \
                        (len(line) > 0) and (not line.isspace())]

    matches = []
    previous_end = None
    for i, line in enumerate(lines):

        # line pattern matching any white
        line_pattern = "\s*".join([re.escape(tok) for tok in line.split()])

        # search for line pattern
        line_match = re.search(line_pattern, target, re.MULTILINE)

        if not line_match:
            for c in target[:1000]:
                print(c, ord(c))

            print('='*88)
            for c in line:
                print(c, ord(c))


            z = sldkjf

        # make sure a matches found
        assert bool(line_match), '''
        Sent = {}\n\n{}\n
        Sent pat = {}\n\n{}\n
        <Text =>\n{}\n\n{}\n'''.format( \
             repr(line), '='*72, line_pattern, '='*72, target, '='*72)

        # get start and end of match
        start = line_match.start()
        end = line_match.end()

        # check if only whitespace exists between previous and current match
        whitespace_only = (previous_end is not None) and \
                         target[previous_end:start].isspace()

        # only whitespace, so adjust previous
        if whitespace_only:
            matches[-1] = (matches[-1][0], end)

        # non white space so create new
        else:
            matches.append((start, end))

        # save end for next iteration
        previous_end = end

    matching_text = "\n".join([target[start:end] for (start, end) in matches])


    return matching_text
