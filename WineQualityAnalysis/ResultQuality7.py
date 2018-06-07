import os

def readSeven(filename):
    """Read the occurrences of each quality value of wine; Return a list of outliers"""
    with open(filename, 'r') as f:
        next(f)  # Skip format line
        for line in f:
            if line[-4] == "7":
                appendLine(line)

def appendLine(line):
    """Append the supplied string to file"""
    snOut = open("ResultsSeven.csv", "a")
    snOut.write(line)
    snOut.close()

try:
    os.remove("ResultsSeven.csv")  # Delete old results file since we are appending
except FileNotFoundError:
    pass

readSeven("training.csv")
readSeven("testing.csv")

