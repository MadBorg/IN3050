import random

# from geneticAlgorithm import * 
import geneticAlgorithm as GA

def test_crash():
# Test if crash
    a = b = None
    n = 1000

    for k in range(n):
        a = [i for i in range(1,random.randint(10, 1000))]
        b = a[:]
        while a == b:
            random.shuffle(b)
        # print("here")
        tmp = GA.pmx(a,b)
        if None in tmp:
            print(f"Err: len(a):{len(a)}, len(b):{len(b)}, k:{k}\n    a:{a}\n    b:{b}\n    tmp:{tmp}")
        if k % (n//10) == 0:
            print(f"{k}")

def test_pmx():
    c1 = 3
    c2 = 7
    P1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    P2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    expectedOffspring = [9, 3, 2, 4, 5, 6, 7, 1, 8]
    calculatedOffspring = GA.pmx(P1, P2, c1, c2)
    msg = f"expected: {expectedOffspring}, calculated: {calculatedOffspring}"
    assert calculatedOffspring == expectedOffspring, msg

    for i in range(10,100):
        P1 = P2 = [0]
        for _ in range(100):
            P1 = [i for i in range(10, 100)]
            P2 = [i for i in range(10, 100)]
            while P1 == P2:
                random.shuffle( P1 )
                random.shuffle( P2 )
            GA.pmx(P1, P2)

    print(calculatedOffspring)


def test_inversionMutation_consistency():
    n = int( 1e5 )
    InListShort = list("abcdef")
    InListLong = [i for i in range(0,100)]
    outListShort = InListShort[:]
    outListLong = InListLong[:]
    for _ in range(n):
        GA.inverstionMutation(outListShort)
        GA.inverstionMutation(outListLong)
        assert len(outListShort) == len(InListShort)
        assert len(outListLong) == len(outListLong)


if __name__ == "__main__":
    test_crash()
    test_inversionMutation_consistency()
    test_pmx()
