from numpy import random


def calculateGrade(question):
    grade = 4
    question_lower = question.lower()

    if 'battery' not in question_lower and 'voltage source' not in question_lower:
        grade = grade - 1

    if 'wire' not in question_lower and 'conductor' not in question_lower and 'closed circuit' not in question_lower:
        grade = grade - 1

    if 'resistor' not in question_lower and 'load' not in question_lower and 'light bulb' not in question_lower:
        grade = grade - 1

    return grade

