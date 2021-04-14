
from collections import OrderedDict, Counter
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pylcs


import config.constants as constants
from config.constants import DOC_LABELS, SUBTYPE_A, SUBTYPE_B
from corpus.tokenization import get_tokenizer
TOKENIZER = get_tokenizer()



def PRF(df):

    df[constants.P] = df[constants.TP]/(df[constants.NP].astype(float))
    df[constants.R] = df[constants.TP]/(df[constants.NT].astype(float))
    df[constants.F1] = 2*df[constants.P]*df[constants.R]/(df[constants.P] + df[constants.R])

    return df


def prf1(df):

    df[constants.FN] = df[constants.NT] - df[constants.TP]
    df[constants.FP] = df[constants.NP] - df[constants.TP]
    df[constants.P] = df[constants.TP].astype(float)/(df[constants.NP].astype(float))
    df[constants.R] = df[constants.TP].astype(float)/(df[constants.NT].astype(float))
    df[constants.F1] = 2*df[constants.P]*df[constants.R]/(df[constants.P] + df[constants.R])

    return df


def has_overlap(a, b):


    A = set(range(*a))
    B = set(range(*b))

    return len(A.intersection(B)) > 0




def entity_indices_exact(t, p):

    # indices match?
    indices_match = t.indices() == p.indices()

    return int(indices_match)

def entity_indices_overlap(t, p):

    # indices overlap?
    indices_match = has_overlap(t.indices(), p.indices())

    return int(indices_match)


def entity_indices_partial(t, p):


    # find overlapping character indices
    ol_start = max(t.start, p.start)
    ol_end   = min(t.end,   p.end)
    ol_n = ol_end - ol_start

    # no overlap
    if ol_n <= 0:
        return 0

    # at least one character
    else:

        crop = lambda text, start, ol_start, ol_n: text[ol_start-start:ol_start-start+ol_n]

        t_text = crop(t.text, t.start, ol_start, ol_n)
        p_text = crop(p.text, p.start, ol_start, ol_n)

        assert t_text == p_text, f'''"{t_text}" VS "{p_text}"'''

        tokens = list(TOKENIZER(t_text))

        return len(tokens)


#def entity_equiv(t, p, exact=True, subtype=True):
def entity_equiv(t, p, match_type="exact", subtype=True):


    # type match?
    type_match = t.type_ == p.type_

    # sub type mach?
    if subtype:
        type_match = type_match and (t.subtype == p.subtype)

    type_match = int(type_match)


    #if exact:
    if match_type == "exact":
        indices_match = entity_indices_exact(t, p)

    elif match_type == "overlap":
        indices_match = entity_indices_overlap(t, p)

    elif match_type == "partial":
        indices_match = entity_indices_partial(t, p)
    else:
        raise ValueError(f"invalid match_type: {match_type}")

    return type_match*indices_match



def compare_doc_labels(T, P, out_type='DataFrame'):
    '''
    Compare entities, only requiring overlap for corpus (e.g. list of documents)
    '''

    # initialize counters
    count_true = Counter()
    count_predict = Counter()
    count_match = Counter()

    # iterate over documents
    assert len(T) == len(P)
    for t, p in zip(T, P):
        for k in t:
            count_true[(k, t[k])] += 1
            count_predict[(k, p[k])] += 1

            if t[k] == p[k]:
                count_match[(k, t[k])] += 1

    if out_type == "DataFrame":

        x = []
        keys = set(count_true.keys()).union(set(count_predict.keys()))
        for k in keys:
            x.append(list(k) + [count_true[k], count_predict[k], count_match[k]])


        fields = [constants.TYPE, constants.SUBTYPE]
        columns = fields + [constants.NT, constants.NP, constants.TP]

        df = pd.DataFrame(x, columns=columns)

        df = prf1(df)

        df = df.sort_values(fields)

        return df

    elif out_type == "Counter":
        return (count_true, count_predict, count_match)
    else:
        ValueError("Invalid output type")



def compare_sent_labels(T, P, out_type='DataFrame'):
    '''
    Compare entities, only requiring overlap for corpus (e.g. list of documents)
    '''

    # initialize counters
    count_true = Counter()
    count_predict = Counter()
    count_match = Counter()

    # iterate over documents
    assert len(T) == len(P)


    # iterate over documents
    for t, p in zip(T, P):

        # iterate over document level types
        for doc_label, subtype_labels in t.items():

            # iterate over subtype combinations
            for (subtype_a, subtype_b) in subtype_labels:
                c = (subtype_a, subtype_b)
                k = (doc_label, subtype_a, subtype_b)

                t_ = t[doc_label][c]
                p_ = p[doc_label][c]

                count_true[k]    += sum(t_)
                count_predict[k] += sum(p_)
                count_match[k]   += sum([int(a == b)*a for a, b in zip(t_, p_)])


    if out_type == "DataFrame":

        x = []
        keys = set(count_true.keys()).union(set(count_predict.keys()))
        for k in keys:
            x.append(list(k) + [count_true[k], count_predict[k], count_match[k]])


        fields = [DOC_LABELS, SUBTYPE_A, SUBTYPE_B]
        columns = fields + [constants.NT, constants.NP, constants.TP]

        df = pd.DataFrame(x, columns=columns)

        df = prf1(df)

        df = df.sort_values(fields)

        return df

    elif out_type == "Counter":
        return (count_true, count_predict, count_match)
    else:
        ValueError("Invalid output type")


def entity_hist(X, match_type="exact", subtype=True):

    counter = Counter()
    for x in X:

        if subtype:
            k = (x.type_, x.subtype)
        else:
            k = (x.type_,)

        if match_type in ["exact", "overlap"]:
            c = 1
        elif match_type in ["partial"]:
            c = len(list(TOKENIZER(x.text)))
        else:
            raise ValueError("invalid match_type")

        counter[k] += c

    return counter

#def compare_entities_doc(T, P, exact=True, subtype=True):
def compare_entities_doc(T, P, match_type="exact", subtype=True):

    # entity count, truth
    count_true = entity_hist(T, match_type=match_type, subtype=subtype)

    # entity count, prediction
    count_predict = entity_hist(P, match_type=match_type, subtype=subtype)

    # entity count, correct
    p_found = set([])
    count_match = Counter()
    for t in T:
        for i, p in enumerate(P):

            # only count if all are true
            match_score = entity_equiv(t, p, match_type=match_type, subtype=subtype)

            if (match_score > 0) and (i not in p_found):

                if subtype:
                    k = (t.type_, t.subtype)
                else:
                    k = (t.type_,)

                count_match[k] += match_score

                p_found.add(i)
                break

    return (count_true, count_predict, count_match)


#def compare_entities(T, P, exact=True, subtype=True, out_type='DataFrame'):
def compare_entities(T, P, match_type="exact", subtype=True, out_type='DataFrame'):
    '''
    Compare entities, only requiring overlap for corpus (e.g. list of documents)
    '''

    # initialize counters
    count_true = Counter()
    count_predict = Counter()
    count_match = Counter()

    # iterate over documents
    assert len(T) == len(P)
    for t, p in zip(T, P):
        c_t, c_p, c_m = compare_entities_doc(t, p, match_type=match_type, subtype=subtype)
        count_true += c_t
        count_predict += c_p
        count_match += c_m


    if out_type == "DataFrame":

        x = []
        keys = set(count_true.keys()).union(set(count_predict.keys()))
        for k in keys:
            x.append(list(k) + [count_true[k], count_predict[k], count_match[k]])

        if subtype:
            fields = [constants.TYPE, constants.SUBTYPE]
        else:
            fields = [constants.TYPE]

        columns = fields + [constants.NT, constants.NP, constants.TP]

        df = pd.DataFrame(x, columns=columns)

        df = prf1(df)

        df = df.sort_values(fields)

        return df

    elif out_type == "Counter":
        return (count_true, count_predict, count_match)
    else:
        ValueError("Invalid output type")







def compare_relations(T, P, \
                    match_type = "exact",
                    subtype = False, out_type = 'DataFrame'):

    # initialize counters
    count_true = Counter()
    count_predict = Counter()
    count_match = Counter()

    # iterate over documents
    assert len(T) == len(P)
    for t, p in zip(T, P):
        c_t, c_p, c_m = compare_relations_doc(t, p, \
                                match_type = match_type,
                                subtype = subtype)
        count_true += c_t
        count_predict += c_p
        count_match += c_m

    if out_type == "DataFrame":

        x = []
        keys = set(count_true.keys()).union(set(count_predict.keys()))
        for k in keys:
            a, b, role = k
            x.append((a, b, role, count_true[k], count_predict[k], count_match[k]))

        columns = [constants.ARG_1, constants.ARG_2, constants.ROLE, \
                   constants.NT, constants.NP, constants.TP]
        df = pd.DataFrame(x, columns=columns)
        df = prf1(df)

        df = df.sort_values([constants.ARG_1, constants.ARG_2, constants.ROLE])

        return df

    elif out_type == "Counter":
        return (count_true, count_predict, count_match)
    else:
        ValueError("Invalid output type")

def relation_hist(X):
    '''
    Create relation histogram
    '''
    return Counter([(x.entity_a.type_, x.entity_b.type_, x.role) for x in X])


def compare_relations_doc(T, P, match_type="exact", subtype=False):

    assert match_type in ["exact", "overlap"]

    # entity count, truth
    count_true = relation_hist(T)

    # entity count, prediction
    count_predict = relation_hist(P)

    # entity count, correct
    count_match = Counter()
    for t in T:
        for p in P:
            a = entity_equiv(t.entity_a, p.entity_a, \
                                match_type = match_type,
                                subtype = subtype)
            b = entity_equiv(t.entity_b, p.entity_b, \
                                match_type = match_type,
                                subtype = subtype)

            # only count if all are true
            if a and b and (t.role == p.role):
                count_match[(t.entity_a.type_, t.entity_b.type_, t.role)] += 1
                break

    return (count_true, count_predict, count_match)
