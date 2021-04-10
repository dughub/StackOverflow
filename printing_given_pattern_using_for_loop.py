# Printing given pattern using for loop
# https://stackoverflow.com/questions/67034346/printing-given-pattern-using-for-loop

# Assuming the constraint of using pure python 3.x only
# N.B. Works only for single digit numbers, could add zero padded formatting
# More efficient itertools / recursive function methods could be used

def print_pyramid(n=4):
    """Print a symmetrical pyramid of numbers descending from n"""

    # Calculate bottom half of grid
    bottom = []
    for j in range(n):
        row = [max(i, j + 1) for i in range(1, n + 1)]
        row = row[::-1][:-1] + row
        bottom.append(row)

    # Invert bottom to get top
    rows = bottom[::-1][:-1] + bottom

    # Print formatted
    for row in rows:
        row_str = [str(i) for i in row]
        print(f"{' '.join(row_str)}")


print_pyramid(9)