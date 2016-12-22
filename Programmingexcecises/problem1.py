__author__ = 'chaitanya'
"""
Below program will find maximum value of multiplication of four maximum distinct palindrome,whose sum is also a palindrome,
 numbers from 10 and 1000 and displays it along with the numbers
...
"""


def is_palindrome(val):
    """This function take a val between 10 and 1000
        and checks if it is a palindrome or not.
    """

    """Input:an integer between 10 and 1000

        Output:boolean true or false
    """

    res = 0
    temp = val

    while val > 0:
        rem = val % 10
        val /= 10
        res = res * 10 + rem
    if res == temp:
        return True
    else:
        return False


def max_mul(low, high):
    """This function take low and high value as (10, 1000) and
        displays four distinct max palinrome numbers and their product value
    """

    """Input:low value(10) and high value (1000) of the range
    """

    """Output:Displays maximum product of 4 distinct palindromes with their product
    """

    counter = high - 1
    nums = []
    max_value = 0
    list1 = []
    # Below code block will run through  four for loops times and finds  four distinct palindrome values ,
    # whose summation is a palindrome, and finds largest multiplication of  four such values
    for i in range(11, 1000):
        if is_palindrome(i):
        #    list1.append(i)
            for j in range(i+1, 1000):
                if is_palindrome(j):
                #     list1.append(j)
                    for k in range(j+1, 1000):
                        if is_palindrome(k):
                        #     list1.append(k)
                            for l in range(k+1, 1000):
                                if is_palindrome(l):
                                #     list1.append(l)
                                    if is_palindrome((i+j+k+l)):
                                        #print (i+j+k+l)
                                        mul = i * j * k * l
                                        if max_value < mul:
                                            max_value = mul
                                            del nums[:]
                                            nums.append(i)
                                            nums.append(j)
                                            nums.append(k)
                                            nums.append(l)
                                    #else:
                                    #del list1[:]

    print "Four maximum palindrome numbers are", nums
    product = nums[0] * nums[1] * nums[2] * nums[3]
    print "Their Product is :", product


max_mul(10, 1000)
