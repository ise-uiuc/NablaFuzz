import os
from typing import List

from classes.oracles import ResultType

def count_results(results: List["ResultType"], keys: List[str] = []):
    """
    Count the number of BUG, FAIL and SUCCESS
    Return (#bug, #fail, #success)
    """
    res_type_count = {}
    for t in ResultType:
        res_type_count[str(t).replace("ResultType.", "")] = 0

    for result in results:
        res_type_count[str(result).replace("ResultType.", "")] += 1

    if len(keys) == 0:
        return res_type_count
    else:
        res = {}
        for k in keys:
            res[k] = res_type_count[k]
        return res
