import math

# Constant for pi
PI = 3.14159


def calculate_area(radius):
    pi = PI
    return pi * (radius**2)


def calculate_circumference(radius):
    pi = PI
    return 2 * pi * radius


def calculate_volume(radius):
    return (4 / 3) * PI * (radius**3)
