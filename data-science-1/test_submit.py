from main import *


def test_q1():
    result = q1()

    assert result == (0.31 , -0.01 , -0.316)


def test_q2():
    result = q2()

    assert round(result, 3) == 0.684


def test_q3():
    result = q3()

    assert result == (0.106, 0.22)


def test_q4():
    result = q4()

    assert result == (0.806, 0.911, 0.959)


def test_q5():
    result = q5()

    assert result == (0.027, 0.04, -0.004)
