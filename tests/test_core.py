import axiom_cpp

def test_hello():
    result = axiom_cpp.hello()
    assert result == "Hello from Axiom C++ core!", f"Unexpected result: {result}"

if __name__ == "__main__":
    test_hello()
    print("test_hello passed!")
