from enum import IntEnum


class ResType(IntEnum):
    RANDOM = 1
    STATUS = 2
    VALUE = 3
    REV_STATUS = 4
    REV_VALUE = 5
    FWD_STATUS = 6
    FWD_VALUE = 7
    REV_FWD_GRAD = 8
    ND_GRAD = 9
    PASS = 10
    CRASH = 11
    SKIP = 12
    DIRECT_CRASH = 13
    REV_CRASH = 14
    FWD_CRASH = 15
    ND_CRASH = 16
    NAN = 17
    ND_FAIL = 18


if __name__ == "__main__":
    a = [str(i).replace("ResType.", "") for i in ResType]
    print(", ".join(a))
