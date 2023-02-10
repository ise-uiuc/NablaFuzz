import argparse
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import json
import random
from secrets import choice
from pathlib import Path
import sys
from os.path import join
from termcolor import colored
import configparser
from classes.oracles import ResultType
from classes.database import TFDatabase
from utils.printer import dump_data
from utils.loader import load_data
from constant.keys import *
from classes.library_def import tf_lib_def
from classes.tf_api import TFAPI
import numpy as np
from classes.tf_gradoracles import IntegratedGradientOracleTF

MAX_SEED_REC = 100
       
def get_seed(api_name, verbose=False):
    
    db = TFDatabase
    records = db.get_all_records(api_name)

    if len(records) == 0:
        return None

    if len(records) > MAX_SEED_REC:
        from random import sample
        records = sample(records, MAX_SEED_REC)

    api = TFAPI(api_name)
    if api.api_def == None: 
        print("Error loading API definition for ", api_name)
        return None
    
    oracle:IntegratedGradientOracleTF = IntegratedGradientOracleTF()
    mutants = {}
    record_set = set()
    for k, record in enumerate(records):
        record_str = json.dumps(record, sort_keys=True)
        if record_str in record_set: continue
        
        load_record_status = api.get_invocation(record)
        if not load_record_status:
            continue

        arg_code, func_exec_def_code, tensor_input_names = oracle.api_to_code(api, verbose=verbose)
        code = "\n".join([arg_code, func_exec_def_code])
        api_record_dict = api.to_record()
        
        random_seed = random.randint(7, 100007)
        mutants[k] = {
            "random_seed": random_seed,
            "input_names": tensor_input_names, 
            "code": code,
            "record": api_record_dict
        }
    return mutants

def get_mutant(api_name, num, verbose=False):
    
    db = TFDatabase
    records = db.get_all_records(api_name)

    if len(records) == 0:
        return None

    if len(records) > num:
        from random import sample
        records = sample(records, num)
    else:
        new_records = random.choices(records, k=num)
        records = new_records

    api = TFAPI(api_name)
    if api.api_def == None: 
        return None
    
    oracle:IntegratedGradientOracleTF = IntegratedGradientOracleTF()
    mutants = {}
    record_set = set()
    for k, record in enumerate(records):
        record_str = json.dumps(record, sort_keys=True)
        
        load_record_status = api.get_invocation(record)
        if not load_record_status:
            continue

        api.mutate()
        arg_code, func_exec_def_code, tensor_input_names = oracle.api_to_code(api, verbose=verbose)
        code = "\n".join([arg_code, func_exec_def_code])
        api_record_dict = api.to_record()
        
        random_seed = random.randint(7, 100007)
        mutants[k] = {
            "random_seed": random_seed,
            "input_names": tensor_input_names,
            "code": code,
            "record": api_record_dict
        }
    return mutants
    
def dump_mutant(api_name, num, out_dir):
    mutants = get_mutant(api_name, num)
    with open(os.path.join(out_dir, "{}.json".format(api_name)), "w") as f:
        f.write(json.dumps(mutants))

def fuzz_all(num, apis, expr_dir, log_dir):
        
    expr_dir = Path(expr_dir)
    
    # dump seed
    out_dir = expr_dir / "seed"
    os.makedirs(out_dir, exist_ok=True)
    for api_name in apis:
        seeds = get_seed(api_name)
        with open(os.path.join(out_dir, "{}.json".format(api_name)), "w") as f:
            f.write(json.dumps(seeds))

    # dump mutant
    out_dir = expr_dir / "mutant"
    t1 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    for api_name in apis:
        dump_mutant(api_name, num, out_dir)

    t2 = time.time()
    with open(os.path.join(log_dir, "time.log"), "a") as f:
        f.write("Fuzz - dump TensorFlow mutant: {} s\n".format(t2-t1))

