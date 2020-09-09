# This program gets a list of points as observation and finds the best line which represents
# that dataset, i. e., the regression line

# Making a list of observations in the following format:
# datapoints = [(1, 4), (3,5)]

datapoints = []
datapoint = None
while datapoint != 'Done':
    # input example: 2, 3
    datapoint = input("Enter the observation or enter Done when you are done:\ninput example: 2, 3\n")
    if datapoint != 'Done':
        commapos = datapoint.find(',')
        x = int(datapoint[0:commapos])
        y = int(datapoint[commapos+2:])
        datapoints.append((x, y))

# Assuming y = m*x + b as the regression line, this function takes the arguments m, b, and x and
# returns the amount of y on the line

def get_y(m, b, x):
    y = m*x + b
    return y

# This function calculates the total square of errors between the datapoints and a given line
# using the m, and b arguments to form the line and datapoint to use the x and y values. Note that
# datapoint needs to be in the following format:
# (x_value, y_value)

def calculate_error(m, b, datapoint):
    x_value = datapoint[0]
    y_value = datapoint[1]
    #print('y_values is: ', y_value)

    y = get_y(m ,b, x_value)
    #print('y for line is: ', y)

    distance = abs(y - y_value)
    #print('distance is: ', distance)
    distance_squared = distance**2
    #print('distance squared is: ', distance_squared)
    return distance_squared

# This function uses the calculate_error function to calculate the total error of all datapoints
# for a given line with m, and b and return the result as a list in the following format:
# lines = [(m, b, total_error)]

def calculate_total_error(m, b, datapoints):
    datapoint = None
    total_error = 0
    for datapoint in datapoints:
        #print('datapoint is: ', datapoint)
        total_error += calculate_error(m, b, datapoint)
        #print('total_error is: ', total_error)
    #print('tital_error is: ', total_error)
    return total_error

# Generating two lists for different vlaues for m and b. This part can be changed for higher precision

# Possible values for m would be -10.00, -9.99, -9.98, ..., 9.98, 9.99, 10.00
possible_ms = [m*0.01 for m in range(-1000, 1001)]
#possible_ms = [m for m in range(1, 4)]


# Possible values for b would be -10.00, -9.99, -9.98, ..., 9.98, 9.99, 10.00
possible_bs = [b*0.01 for b in range(-1000, 1001)]
#possible_bs = [b for b in range(0, 3)]


# This function takes the possible values for m and b, and the datapoints for an observation, and uses
# the calculate_total_error function to calculate the total error for all possible lines (with all possible
# combinations of m and b) for the datapoints and eventually returns the best line with the least error
# in the following format:
# [(m, b, total_error)]

def line_finder(m_list, b_list, datapoints):
    total_errors_list = []
    for m in m_list:
        #print('m is: ', m)
        for b in b_list:
            #print('b is: ', b)
            total_error = calculate_total_error(m, b, datapoints)
            total_errors_list.append((m, b, total_error))
            #print('total_errors_list is: ', total_errors_list)
    smallest_error = 10**10
    #print(smallest_error)
    best_line = None
    for possible_line in total_errors_list:
        #print('possible line is: ', possible_line)
        if possible_line[2] < smallest_error:
            smallest_error = possible_line[2]
            #print('True')
            #print('possible line[2] is: ', possible_line[2])
            best_line = possible_line
            #print('********** best_line is: ', best_line)
        else:
            #print('False')
            continue
    #print('Final best line is: ', best_line)
    return best_line

# Finding out the regression line
regression_line = line_finder(possible_ms, possible_bs, datapoints)
#print('regression_line is: ', regression_line)
best_m = int(regression_line[0])
best_b = int(regression_line[1])
print('The regression line is: \ny=', str(best_m), '* x +', str(best_b))

print('\nThanks for reviewing')

# Thanks for reviewing
