import logging
import os
from collections import OrderedDict, Counter
import pandas as pd


class Scorer(object):

    def __init__(self):
            pass


    def compare(self, T, P):
        return None

    def combine(self, dfs):
        return None


    def fit(self, T, P, params=None, path=None):
        '''
        Score predictions

        Parameters
        ----------
        T = truth
        P = predictions

        '''

        # Check sentence count
        len_check(T, P)

        # Get counts
        dfs = self.compare(T, P)

        if not isinstance(dfs, dict):
            dfs = OrderedDict([('default', dfs)])


        for k, df in dfs.items():
            logging.info('\n\n{}\n{}'.format(k, df))

        # Include each parameter in data frame
        if params is not None:
            dfs = {k:add_params_to_df(df, params) for k, df in dfs.items()}

        if path is not None:

            for k, df in dfs.items():

                if len(dfs) == 1:
                    f = os.path.join(path, f"scores.csv")
                else:
                    f = os.path.join(path, f"scores_{k}.csv")
                df.to_csv(f)

        return dfs




    def combine_cv(self, dfs, path=None):


        dfs = self.combine(dfs)

        if path is not None:
            for k, df in dfs.items():
                if len(dfs) == 1:
                    f = os.path.join(path, f"scores.csv")
                else:
                    f = os.path.join(path, f"scores_{k}.csv")
                df.to_csv(f)

        return dfs




def len_check(x, y):
    assert len(x) == len(y), "length mismatch: {} vs {}".format(len(x), len(y))

def add_params_to_df(df, params):

    # Loop on Level 1
    for p1, v1 in params.items():

        # Level 1 as dictionary
        if isinstance(v1, dict):

            # Loop on level 2
            for p2, v2 in v1.items():

                # Level 2 as dictionary
                if isinstance(v2, dict):

                    # Loop on level 3
                    for p3, v3 in v2.items():

                        # Level 3 is dictionary
                        if isinstance(v3, dict):
                            df[str((p1, p2, p3))] = str(v3)

                        # Level 3 is not dict, list, or array
                        elif not isinstance(v3, (list, np.ndarray)):
                            df[str((p1, p2, p3))] = v3

                # Level 2 is not dict, list, or array
                elif not isinstance(v2, (list, np.ndarray)):
                    df[str((p1, p2))] = v2

        # Level 1 is not dict, list, or array
        elif not isinstance(v1, (list, np.ndarray)):
            df[p1] = v1



    return df
