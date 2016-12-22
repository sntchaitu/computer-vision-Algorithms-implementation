__author__ = 'chaitanya'
# The program will generate all perfect number below 1 and 10000 and prints those numbers

def is_perfect_number(val):

    """This function  will check if given number is a perfect number or not
    """
    """
        input:an integer

        output:True or False

    """

    temp = 1
    sum = 0
    while temp < val:
        # starts from 1 until the val and checks if each number divides
        if val % temp == 0:
            sum += temp
        temp += 1
    if sum == val:
        return True
    else:
        return False


def pf_generator(low,high):
    """This function  will generate all perfect number below 10000 and prints those numbers
    """
    """
        input:low and high value of the range

        output: prints all the perfect numbers in the range

    """
    counter = low
    nums = []
    # starts from 1 until the high  and checks if each number is a perfect number
    while counter < high:
        if is_perfect_number(counter):
            nums.append(counter)
        counter+=1
    print 'numbers are', nums

pf_generator(1,10000)