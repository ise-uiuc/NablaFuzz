def handle_params(params):
    del params['_id']
    arg_list = []
    kwarg_dict = {}
    input_arg_list = []
    input_kwarg_dict = {}
    for item in params.items():
        if item[0].startswith('init_'):
            if item[0].startswith('init_parameter_'):
                input_arg_list.append(item[1])
            else:
                item_name = item[0].replace('init_','')
                input_kwarg_dict[item_name] = item[1]

        else:
            if item[0].startswith('parameter_'):
                arg_list.append(item[1])
            else:
                kwarg_dict[item[0]] = item[1]
    return arg_list, kwarg_dict, input_arg_list, input_kwarg_dict