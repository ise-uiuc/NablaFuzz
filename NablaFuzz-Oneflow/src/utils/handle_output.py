import oneflow

def output_handler(output):
    out_type = type(output).__name__
    if out_type == 'Tensor' or out_type == 'Parameter':
        sum = oneflow.sum(output)
    elif out_type == 'tuple':
        sum = 0
        for t in output:
            sum += oneflow.sum(t)
    return sum