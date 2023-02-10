from functools import wraps
import oneflow
import pymongo
import sys



def mongo_save(collection_name, data):
    host = "localhost"
    port = 27017
    mongo_db = "autodiff-oneflow-5"
    client = pymongo.MongoClient(host=host, port=port)
    db = client[mongo_db]
    collection = db[collection_name]
    if collection.count_documents({}) >= 5000:
        pass
        #print(collection_name,': too many instances (exceeds 200)')
    else:
        try:
            collection.update_one(data,{'$set':data},upsert=True)
            print('insertion succeed')
        except Exception as e:
            print('insertion failed:',collection_name, data)
            print(e)
    


def decorate_func(func,line):
    @wraps(func)
    def wrapper(*args, **kwargs):
        current_dict = {}
        def parse_arg(arg, current_dict, arg_key=None): #command: what do you want: type, shape...
            if arg_key == None:
                para_name = 'parameter_'+str(len(current_dict))
                para_dict = get_para_dict(arg)
            else:
                para_name = str(arg_key)
                #print(arg_key)
                para_dict = get_para_dict(arg)
            current_dict[para_name] = para_dict
            return current_dict
        
        def get_para_dict(arg):
            arg_type = type(arg).__name__
            if arg_type == 'Tensor' or arg_type == 'Parameter':
                arg_shape = list(arg.shape)
                arg_dtype = arg.dtype
                return {'type':arg_type, 'shape':arg_shape, 'dtype': str(arg_dtype)}
            elif arg_type == 'ndarray':
                arg_shape = list(arg.shape)
                arg_dtype = arg.dtype
                return {'type':arg_type, 'shape':arg_shape, 'dtype': str(arg_dtype)}
            elif arg_type == 'Size':
                return list(arg)
            elif arg_type == 'list':
                new_list = []
                for x in arg:
                    element_arg_type = type(x).__name__
                    if element_arg_type == 'Tensor' or element_arg_type == 'Parameter':
                        new_list.append({'type':element_arg_type, 'shape':list(x.shape), 'dtype': str(x.dtype)})
                    else:
                        new_list.append(x)
                return new_list
            elif arg_type in ['str', 'int', 'float', 'complex', 'tuple','range','dict','set','frozenset','bool','bytes','bytearray','memoryview']:
                return arg
            else:
                return {'type': arg_type, 'content': str(arg)}
            
        
        #print(output)
        for arg in args:
            current_dict = parse_arg(arg, current_dict)
        for idx, kwarg in enumerate(kwargs):
            current_dict = parse_arg(kwargs[kwarg], current_dict, arg_key=kwarg)
        
        mongo_save(line, current_dict)
        output = func(*args, **kwargs)
        params_dict = {} #avoid duplicate key and accumulation
        return output
    return wrapper



def decorate_class(class_name, line):
    def get_para_dict_class(arg):
        arg_type = type(arg).__name__
        if arg_type == 'Tensor' or arg_type == 'Parameter':
            arg_shape = list(arg.shape)
            arg_dtype = arg.dtype
            return {'type':arg_type, 'shape':arg_shape, 'dtype': str(arg_dtype)}
        elif arg_type == 'ndarray':
            arg_shape = list(arg.shape)
            arg_dtype = arg.dtype
            return {'type':arg_type, 'shape':arg_shape, 'dtype': str(arg_dtype)}
        elif arg_type == 'Size':
            return list(arg)
        elif arg_type == 'list':
            new_list = []
            for x in arg:
                element_arg_type = type(x).__name__
                if element_arg_type == 'Tensor' or element_arg_type == 'Parameter':
                    new_list.append({'type':element_arg_type, 'shape':list(x.shape), 'dtype': str(x.dtype)})
                else:
                    new_list.append(x)
            return new_list 
        elif arg_type in ['str', 'int', 'float', 'complex', 'tuple','range','dict','set','frozenset','bool','bytes','bytearray','memoryview']:
            return arg
        else:
            return {'type': arg_type, 'content': str(arg)}

    def parse_arg_class(arg, current_dict, arg_key=None, is_init=False): #command: what do you want: type, shape...
        if arg_key == None:
            if is_init:
                para_name = 'init_parameter_' + str(len(current_dict))
            else:
                para_name = 'parameter_' +  str(len(current_dict))
            para_dict = get_para_dict_class(arg)
        else:
            if is_init:
                para_name = 'init_' + str(arg_key)
            else:
                para_name = str(arg_key)
            para_dict = get_para_dict_class(arg)
        current_dict[para_name] = para_dict
        return current_dict

    if not hasattr(class_name, "__call__"):
        pass
    if class_name.__init__ == object.__init__:
        pass
    params_dict = {}
    
    old_init = class_name.__init__
    old_call = class_name.__call__

    def new_init(self, *args, **kwargs):
        nonlocal params_dict

        params_dict = {}
        #print(args)
        for arg in args:
            params_dict = parse_arg_class(arg, params_dict, is_init=True)
        for idx, kwarg in enumerate(kwargs):
            params_dict = parse_arg_class(kwargs[kwarg], params_dict, arg_key=kwarg, is_init=True)
        self.params_dict = params_dict
        old_init(self, *args, **kwargs)    

    def new_call(self, *input_args, **input_kwargs):
        params_dict = self.params_dict.copy()
        #print(line)
        for arg in input_args:
            params_dict = parse_arg_class(arg, params_dict, is_init=False)
        for idx, kwarg in enumerate(input_kwargs):
            params_dict = parse_arg_class(input_kwargs[kwarg], params_dict, arg_key=kwarg, is_init=False)

        mongo_save(line, params_dict)
        output = old_call(self, *input_args, **input_kwargs)
        params_dict = {}  #avoid duplicate key and accumulation
        return output

    setattr(class_name, '__init__', new_init)
    setattr(class_name, '__call__', new_call)

def decorate_all():
    f = open(__file__.replace("__init__.py", "oneflow_apis.txt"))
    lines = f.readlines()
    skipped_list = ['pybind11_type', 'dtype', 'type']

    for line in lines:
        line = line[:-1]
        line_split = line.split(".")
        module_name = ".".join(line_split[:-1])
        func_name = line_split[-1]
        line_skipped_list = ['oneflow.nn.Module']
        if line not in line_skipped_list:
            if hasattr(eval(module_name), func_name):
                if type(eval(line)).__name__ not in skipped_list: #and not func_name.endswith('_'):
                    wrapped_func = decorate_func(eval(line), line)
                    setattr(eval(module_name), func_name, wrapped_func)
                elif type(eval(line)).__name__ == 'type' and func_name not in ['Size']:
                    decorate_class(eval(line), line)

line = 'oneflow.linalg.matrix_norm'
wrapped_func = decorate_func(eval(line),line)
setattr(oneflow.linalg, 'matrix_norm', wrapped_func)

decorate_all()
print('---------------------All decorated---------------------')
