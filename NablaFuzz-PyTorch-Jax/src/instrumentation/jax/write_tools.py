import pymongo

"""
You should configure the database
"""
jax_db = pymongo.MongoClient(host="localhost", port=27017)["jax-test"]


def write_fn(func_name, params, input_signature):
    params = dict(params)
    out_fname = func_name
    if input_signature != None:
        params["input_signature"] = input_signature
    # print(out_fname)
    # print("  "+ str(params))
    jax_db[out_fname].insert_one(params)
