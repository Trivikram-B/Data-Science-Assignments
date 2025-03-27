import numpy as np

# Function to calculate the upper and lower bounds for outliers
def quartile(data, quartile):
    q = np.percentile(data, q=quartile, method='inverted_cdf')
    return q

def find_outliers(data):
    # The 25th (Q1) and 75th (Q3) percentiles
    Q1 = quartile(data, 25)
    Q3 = quartile(data, 75)
    # print(Q1, Q3)
    
    # Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Calculations from formula
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # upper and lower bounds
    print("Lower bound for outliers: " ,lower_bound)
    print("Upper bound for outliers: ", upper_bound)
    
    # Identify outliers
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    if outliers:
        print("Outliers : ",outliers)
    else:
        print("No outliers found.")

input_data = [15, 175, 80, 34, 23, 12, -95, 74, 56, 9, 65]
input_data = sorted(input_data)
# print(input_data)
find_outliers(input_data)
