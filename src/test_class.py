class TestClass:
    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar

    def __repr__(self):
        return repr(
            f"TestClass foo={self.foo} bar={self.bar}"
        )


class TestClass2:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return repr(
            f"TestClass a={self.a} b={self.b}"
        )

def test_func(*args, **kwargs):
    print("args", args)
    print("kwargs", kwargs)

    return "something"