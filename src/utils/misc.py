

from collections import Counter, OrderedDict






def list_to_dict(L, extend_or_append='append'):
    NotImplementedError("rename 'list_to_dict' to 'nest_list'")



def nest_list(L, extend_or_append='append'):

    # Initialize dictionary
    D = OrderedDict([(k,[]) for k, v in L[0].items()])

    # Loop on elements in list
    for l in L:

        # Append values from current element
        for k, v in l.items():

            if extend_or_append == 'append':
                D[k].append(v)
            elif extend_or_append == 'extend':
                D[k].extend(v)
            else:
                raise ValueError('incorrect value for extend_or_append')
    return D



def nest_dict(D):

    Y = []


    lengths = []
    for k, v in D.items():
        lengths.append(len(v))
    assert len(set(lengths)) == 1, "length mismatch: {}".format(lengths)
    length = lengths[0]

    for i in range(length):

        # Loop on dictionary
        d = OrderedDict([(k, v[i]) for k, v in D.items()])

        # Append to list
        Y.append(d)

    return Y
