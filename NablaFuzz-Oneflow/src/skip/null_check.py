def check_null_tensor(args):
    flag = False
    if type(args).__name__ == 'list':
        for arg in args:
            if type(arg).__name__=='dict' and 'type' in arg.keys():
                if arg['type'] in ['Tensor','Parameter']:
                    if arg['shape'] == [] or 0 in arg['shape']:
                        flag = True
                        break
        return flag
    else:
        for item in args.items():
            if type(item[1]).__name__=='dict' and 'type' in item[1].keys():
                if item[1]['type'] in ['Tensor', 'Parameter']:
                    if item[1]['shape']==[] or 0 in item[1]['shape']:
                        flag = True
                        break
        return flag