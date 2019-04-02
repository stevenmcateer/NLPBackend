from numpy import random


def calculateGrade(question, response):
    grade = 4
    res = response.lower()
    print(res)
    if 'battery' not in res and 'voltage source' not in res:
        grade = grade - 1

    if 'wire' not in res and 'conductor' not in res and 'closed circuit' not in res:
        grade = grade - 1

    if 'resistor' not in res and 'load' not in res and 'light bulb' not in res:
        grade = grade - 1

    return grade

