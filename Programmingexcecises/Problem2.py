__author__ = 'chaitanya'


def predict_number(low,high):
    """This function  will search for required number between the range low and high
       and print the chosen number along with the number of search attempts
    """
    """
        input:low and high value of the range

        output: Choose Number (i.e., mid)
                No of search attempts(i.e., count),

    """
    count = 0
    while True:
        # loop until the chosen number is found

        res = (low+high)/2
        print ("Is", res, "the number that you have choosen")
        response = raw_input()

        if response.lower() == 'w':
            count+=1
            break
        elif response.lower() == 'h':
            high = res
        elif response.lower() == 'l':
            low = res
        count+=1

    print "Number is", res,"\n"
    print "No of attempts#", count,

# input method to ask upper bound of the range

X = input('Enter Upper Bound:')

print "Pick a number in your mind"


predict_number(1,X)

