import pymongo
import numpy as np

def get_info(i):
    host = "localhost"
    port = 27017
    mongo_db = "autodiff-oneflow-5"
    client = pymongo.MongoClient(host=host, port=port)
    db = client[mongo_db]
    random_collection_name = db.list_collection_names()[i]
    #random_collection_name = np.random.choice(db.list_collection_names(), 1)[0]
    print(random_collection_name)
    random_collection = db[random_collection_name]
    random_input = np.random.choice(list(random_collection.find({})), 1)[0]
    print(random_input)
    return random_collection_name, random_input