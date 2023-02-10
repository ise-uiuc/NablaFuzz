def check_acceptable(error,error_type):
    acceptable_error = [
        'a leaf Tensor that requires',
        'must be tensor tuple',
        'must be tensor, not NoneType',
        'index out of range',
        'should be true',
        'has no len()',
        'Check failed'
        ]
    acceptable_type = [
        'UnboundLocalError',
        'AssertionError',
        'ValueError',
        'TypeError'
        ]
    for a in acceptable_error:
        if a in error:
            return True
    for a in acceptable_type:
        if error_type == a:
            return True
    return False
