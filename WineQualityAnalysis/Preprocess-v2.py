import random, math


def read_freq(filename):
    """Read the occurrences of each quality value of wine; Return a list of outliers"""
    n = 0  # Total number of data entries
    data = {}  # Dictionary to hold occurrences of each quality
    outliers = []  # Any ratings that are outliers
    with open(filename, 'r') as f:
        next(f)  # Skip format line
        for line in f:
            n += 1
            key = int(str.strip(line)[-1])  # The quality value is the last character
            if key not in data:
                data[key] = 1  # First time that quality is found
            else:
                data[key] += 1  # Increase the quality count

    bound = int(n * 0)  # Data is an outlier if it is less than 3% of total cases
    for key in data:
        if data[key] < bound:
            outliers.append(key)

    # Output the frequencies of each type for reference
    for i in range(0, len(data)):
        low = 10
        for key in data:
            if key < low:
                low = key
        print(low, ":", data[low])
        del data[low]

    return outliers
    # End read_freq()


def combineFile(file1, file2, output):
    """Take two files and combine the data into one file"""
    data = []
    n = 0
    # Read all the data from both files
    with open(file1, "r") as f:
        for line in f:
            data.append(line)
    with open(file2, "r") as f:
        for line in f:
            data.append(line)
    print("\nCombining:", file1, "and", file2)
    print(len(data), "data points detected")
    snOut = open(output, "w")
    for i in range(0, len(data)):
        index = random.randint(0, len(data) - 1)  # Write the elements to file in a random order
        snOut.write(data.pop(index))
        n += 1
    snOut.close()
    print(n, "data points written randomly to", output)  # Print how many data points were written to file
    # End combineFile()


def generateFile(data, filename):
    """Write all the passed in data to a file"""
    snOut = open(filename, "w")
    for i in range(0, len(data)):
        snOut.write(data[i])
    snOut.close()
    # End generateFile()

def floatTransform(data):
    """Transform data from ',' delimited string to floats"""
    mathData = []  # Hold numeric data version of the passed in data
    for line in data:
        nums = []
        values = line.split(',')  # Split line by ',' delimiter
        for v in values:
            nums.append(float(v))  # Convert to float before saving
        mathData.append(nums)  # List of float version of the line
    return mathData
    # End floatTransform()


def stringTransform(mathData):
    """Transform data back into string form with ',' delimiter"""
    stringData = []
    for i in range(0, len(mathData)):
        line = ""
        for j in range(0, len(mathData[i])):
            line += str(mathData[i][j]) + ","  # Add each float term to the string with the delimiter
        line = line[:-1] + "\n"  # Remove extra ',' at the end and add newline
        stringData.append(line)
    return stringData
    # End stringTransform()


def normal(data):
    """Apply a Max/Min normalisation to the data"""
    mathData = floatTransform(data)  # Hold numeric data version of the passed in data

    maximum = [0.0] * (len(mathData[0]) - 1)  # The max value for each of the data values
    minimum = [99999.9] * (len(mathData[0]) - 1)  # The min value for each of the data values

    # Find the max and min for each data value
    for i in range(0, len(mathData)):
        for j in range(0, len(mathData[i]) - 1):
            if mathData[i][j] > maximum[j]:
                maximum[j] = mathData[i][j]
            if mathData[i][j] < minimum[j]:
                minimum[j] = mathData[i][j]

    # X' = (X - Xmin) / (Xmax - Xmin)
    for i in range(0, len(mathData)):
        for j in range(1, len(mathData[i]) - 1):
            mathData[i][j] = (mathData[i][j] - minimum[j]) / (maximum[j] - minimum[j])

    return stringTransform(mathData)
    # End Normal()


def gaussianNormal(data):
    """Apply a Gaussian Normalisation to the data"""
    mathData = floatTransform(data)  # Hold numeric data version of the passed in data

    mean = [0.0] * (len(mathData[0]) - 1)  # The mean for each of the data values
    stdDev = [0.0] * (len(mathData[0]) - 1)  # The standard deviation for each of the data values

    # Calculate the mean for each data value
    for i in range(0, len(mathData)):
        for j in range(0, len(mathData[i]) - 1):
            mean[j] += mathData[i][j]  # Running total sum
    for i in range(0, len(mean)):
        mean[i] = mean[i] / len(mathData)  # Calcuate final mean

    # Calculate the standard deviation for each data value
    for i in range(0, len(mathData)):
        for j in range(0, len(mathData[i]) - 1):
            stdDev[j] += (mathData[i][j] - mean[j])**2  # Running total sum of Std Dev terms
    for i in range(0, len(stdDev)):
        stdDev[i] = math.sqrt(stdDev[i] / len(mathData))  # Calculate final Std Dev

    # Normalize the data: v' = (v - mean) / StdDev
    for i in range(0, len(mathData)):
        for j in range(1, len(mathData[i]) - 1):  # Start at one as first term cannot be normalised
            mathData[i][j] = (mathData[i][j] - mean[j]) / stdDev[j]

    return stringTransform(mathData)
    # End gaussianNormal()


def preprocess(outliers, bit, filename):
    """Preprocess the data set to remove outliers and divide to test and train set"""
    print("\nCreated test and train set for:", filename)
    data = []  # List to hold the data strings
    n = 0  # Counter variable
    with open(filename, 'r') as f:
        next(f)  # Skip format line
        for line in f:
            outlier = False  # Boolean to test if it is an outlier
            string = line.replace(";",",")  # Network is designed to accept comma seperated values
            for o in outliers:
                if int(str.strip(string)[-1]) == o:  # Test if the data string is an outlier
                    outlier = True
                    n += 1
            if not outlier:  # Prune line if it is an outlier
                data.append(str(bit) + "," + string)  # Save all non-outlier lines
    print("Removed", n, "outliers")  # Print the number of outliers removed

    data = normal(data)  # Apply a Magnitude Normalisation to the data
    #data = gaussianNormal(data)  # Apply a Gaussian Normalisation to the data

    # Create filenames for output files
    outFileTrain = filename.replace(".csv", "-train.csv")
    outFileTest = filename.replace(".csv", "-test.csv")

    # Randomly select 20% of the data points to be the test data
    testData = []
    t = int(len(data) * 0.20)
    for i in range(0, t):
        index = random.randint(0, len(data) - 1)
        testData.append(data.pop(index))

    # Create the train and test files
    generateFile(data, outFileTrain)
    generateFile(testData, outFileTest)

    # Print the size of the files
    print("Traning set size:", len(data))
    print("Testing set size:", len(testData))
    # End preprocess()


## MAINLINE ##

print("Original Wine Quality v. Frequency")
print("[Quality] : [Frequency]\n")

print("White Wine\n")
whiteOutliers = read_freq("winequality-white.csv")

print("\nRed Wine\n")
redOutliers = read_freq("winequality-red.csv")

preprocess(whiteOutliers, 1, "winequality-white.csv")
preprocess(redOutliers, 0, "winequality-red.csv")

combineFile("winequality-white-train.csv", "winequality-red-train.csv", "training.csv")
combineFile("winequality-white-test.csv", "winequality-red-test.csv", "testing.csv")

combineFile("Testing.csv", "Training.csv", "Samples.csv")

