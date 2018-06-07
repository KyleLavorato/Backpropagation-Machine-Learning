import random
import math
import time

## Kyle Lavorato | 10141235 ##

################################
#     ANN Class Definition     #
################################

class Network:
    def __init__(self, learningRate, momentum, maxItr, minMSE, inputNum, hiddenNum, outputNum):
        '''Constructor for the network'''
        self.start = time.time()
        self.c = learningRate  # Learning rate
        self.alpha = momentum  # Momentum rate
        self.maxItr = maxItr  # Maximum iterations
        self.minMSE = minMSE   # Minimum acceptable MSE
        self.numIn = inputNum + 1  # Number of input nodes; Add the bias node
        self.numH = hiddenNum  # Number of hidden nodes
        self.numO = outputNum  # Number of output nodes
        self.MSEVal = 1000.0  # The current MSE value
        self.tested = 0  # Number of tested inputs
        self.correct = 0  # Number of correctly classified inputs

        # Declare weight sets
        self.wih = self.makeWeights(self.numIn, self.numH)  # Weights between the input and hidden layers
        self.who = self.makeWeights(self.numH, self.numO)  # Weights between the hidden and output layers
        self.mih = self.makeFloatMatrix(self.numIn, self.numH)  # Previous weight set between input and hidden layers for momentum
        self.mho = self.makeFloatMatrix(self.numH, self.numO)  # Previous weight set between hidden and output layers for momentum

        # Declare activation sets
        # self.ai = [1.0] * self.numIn  # Activation at all input nodes + 1 for bias; Declare as float
        # self.ah = [1.0] * self.numH  # Activation at all hidden nodes; Declare as float
        # self.ao = [1.0] * self.numO  # Activation at all hidden nodes; Declare as float
        self.ai = self.makeFloatArray(self.numIn)  # Activation at all input nodes + 1 for bias; Declare as float
        self.ah = self.makeFloatArray(self.numH)  # Activation at all hidden nodes; Declare as float
        self.ao = self.makeFloatArray(self.numO)  # Activation at all hidden nodes; Declare as float

    def makeWeights(self, x, y):
        '''Generate an array of random values for the weights'''
        # x is the number of inputs that will feed into each next layer node
        # y is the weight set for each node in the next layer
        w = []  # Weights list
        sw = []
        for i in range(0, x):
            for j in range(0, y):
                sw.append(random.uniform(-0.5, 0.5))
            w.append(sw)
        return w

    def makeFloatArray(self, n, fill=1.0):
        '''Generate a list of values'''
        a = []
        for i in range(0, n):
            a.append(fill)
        return a

    def makeFloatMatrix(self, x, y, fill=0.0):
        '''Generate a matrix of values'''
        m = []
        sm = []
        for i in range(0, x):
            for j in range(0, y):
                sm.append(fill)
            m.append(sm)
        return m

    def timer(self):
        '''Calculates the time from creation of the NN to when this function is called'''
        end = time.time()
        elapsed = round(end - self.start, 2)
        print("Total training time of:", elapsed, "seconds")

    def sigmoid(self, x):
        '''The sigmoid function'''
        x = max(x, -500)  # Prevent a math error caused by large negative values
        return (1 / (1 + math.exp(-1 * x)))

    def sigmoidPrime(self, y):
        '''The derivative of the sigmoid function'''
        return y * (1.0 - y)

    def MSE(self, d):
        '''Calculate the MSE of the current iteration'''
        if self.MSEVal == 1000.0:
            self.MSEVal = 0  # Reset if current iteration complete
        error = 0.0
        for i in range(len(d)):
            error += (1 / len(d)) * (d[i] - self.ao[i]) ** 2
        self.MSEVal += error

    def activate(self, dataIn):
        '''Calculate and update the activation for all nodes'''
        # Activation in input layer; All values are Xi
        for i in range(0, self.numIn - 1):  # Skip the last value as Xbias is always one; Set bias to last for ease
            self.ai[i] = dataIn[i]

        # Activation in hidden layer
        for i in range(0, self.numH):
            net = 0.0
            for j in range(0, self.numIn):
                net += self.ai[j] * self.wih[j][i]  # i = hidden node; j = input node
            self.ah[i] = self.sigmoid(net)  # The hidden node's activation is the sigmoid of the input sum

        # Activation in output layer
        for i in range(0, self.numO):
            net = 0.0
            for j in range(0, self.numH):
                net += self.ah[j] * self.who[j][i]  # i = output node; j = hidden node
            self.ao[i] = self.sigmoid(net)  # The output node's activation is the sigmoid of the input sum

    def correction(self, d):
        '''Calculate and apply the adjustments in weights'''
        # d is expected outcome in the form; [0,0,0,0,0,1,0,0,0,0] = 5

        # Calculate the delta term for weight change
        delta_out = []
        for i in range(0, self.numO):
            if d[i] == 1 and self.ao[i] >= 0.95:  # Error is 0 if within the ranges
                e = 0
            elif d[i] == 0 and self.ao[i] <= 0.05:
                e = 0
            else:
                e = d[i] - self.ao[i]  # e = (d - y)
            delta_out.append(e * self.sigmoidPrime(self.ao[i]))  # δo = e * f'(x)

        delta_hidden = []
        for i in range(0, self.numH):
            e = 0.0
            for j in range(0, self.numO):
                e += delta_out[j] * self.who[i][j]
            e *= self.sigmoidPrime(self.ah[i])
            delta_hidden.append(e)  # δh = ∑(δo * Who) * f'(ai)

        # Calculate the change in weight
        for i in range(0, self.numH):  # Output layer adjustment
            for j in range(0, self.numO):
                delta = self.c * delta_out[j] * self.ah[i]  # ∆w = c * δo * Xi
                self.who[i][j] += delta + self.mho[i][j] * self.alpha  # w = w + ∆w^(t+1) + α∆w^(t)
                self.mho[i][j] = delta  # Save ∆w^(t)

        for i in range(0, self.numIn):  # Hidden layer adjustment
            for j in range(0, self.numH):
                delta = self.c * delta_hidden[j] * self.ai[i]  # ∆w = c * δo * Xi
                self.wih[i][j] += delta + self.mih[i][j] * self.alpha  # w = w + ∆w^(t+1) + α∆w^(t)
                self.mih[i][j] = delta  # Save ∆w^(t)

    def train(self, data, validation):
        '''Begins the training process for the network; Called from the main program'''
        iteration = 0
        percent = 0.0
        while self.MSEVal > self.minMSE and iteration < self.maxItr:
            self.MSEVal = 1000.0
            if iteration % 4 == 0:  # Display training progress
                percent = round(iteration / self.maxItr * 100, 1)
                print("Training is", percent, "% complete")
            for i in range(0, len(data)):
                self.activate(data[i])
                self.correction(validation[i])
                self.MSE(validation[i])
            iteration += 1
            self.MSEVal /= len(data)  # Normalize MSE after each iteration
            if iteration % 3 == 0 and self.c > 0.15:  # Reduce learning rate every 3 iterations until min is reached
                self.c /= 1.5
            if self.c < 0.15:
                self.c = 0.15
        # for a in self.ao:
        #     print(a)
        # print("")
        if self.MSEVal < self.minMSE:
            print("\n\nTraining Complete: Error is under acceptable minimum")
        else:
            print("\n\nTraining Complete: Max iterations reached")
            print("Error Value:",self.MSEVal)
        self.timer()

    def test(self, data, validation):
        '''Tests to see if each input is being classified correctly by the network'''
        self.tested = 0
        self.correct = 0
        for j in range(0, len(data)):
            self.activate(data[j])
            value = [i for i,x in enumerate(validation[j]) if x == 1][0]
            # for a in self.ao:
            #     print(a)
            # print("")
            for k in range(0, len(validation[j])):
                if self.ao[k] > 0.90 and validation[j][k] == 1:
                    self.correct += 1
                elif self.ao[k] < 0.10 and validation[j][k] == 0:
                    self.correct += 1
                self.tested += 1
        print(self.correct, "/", self.tested, "numbers classified correctly")

    def classify(self, data):
        '''Write the classification of each input to an output file'''
        with open("class.txt", 'w') as f:
            for i in range(0, len(data)):
                self.activate(data[i])
                yesClass = []
                noClass = []
                for j in range(0, self.numO):
                    if self.ao[j] > 0.90:
                        yesClass.append(str(j))
                    elif self.ao[j] < 0.10:
                        noClass.append(str(j))
                if not yesClass:
                    f.write("Data entry " + str(i+1) + ": Could not be classified\n")
                else:
                    f.write("Data entry " + str(i+1) + ": ")
                    for j in range(0, len(yesClass)):
                        f.write(str(yesClass[j]) + ",")
                    f.write("\n")
                if not noClass:
                    f.write("Data entry " + str(i+1) + ": Could not be classified\n")
                else:
                    f.write("Data entry " + str(i+1) + " is not: ")
                    for j in range(0, len(noClass)):
                        f.write(str(noClass[j]) + ",")
                    f.write("\n")


################################
# Utility Function Definitions #
################################

def readOrigFile(filename):
    '''Read the original form data from file'''
    # Data is stored in a 3D list
    # Index 1: The 32x32 array for one number
    # Index 2: The row of the 32x32 array; eg. [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
    # Index 3: The column of the above row
    data = []  # The full data set
    dataValues = []  # The number that each 32x32 grid is
    with open(filename,'r') as f:
        for i in range(0,21):
            next(f)  # Don't read the first 21 lines
        while True:
            number = []  # The row
            for i in range(0,32):
                lineNums = []  # The column
                line = str.strip(f.readline())
                for s in line:
                    lineNums.append(int(s))
                number.append(lineNums)  # Append the column
            if line == "":
                break
            num = str.strip(f.readline())
            data.append(number)  # Append the row
            dataValues.append(num)
    return data, dataValues


def readModifiedFile(filename):
    '''Read the preprocessed data from file'''
    data = []  # The full data set
    dataValues = []  # The number that each 32x32 grid is
    with open(filename, 'r') as f:
        while True:
            values = []  # The row
            line = str.strip(f.readline())
            if line == "":
                break
            line = line.split(',')
            for s in line:
                values.append(int(s))
            dataValues.append(values[-1])
            values.pop()
            data.append(values)  # Append the row
    return data, dataValues

def preProcess(validation):
    '''Preprocess the validation data to be in a form usable by the Network'''
    set = []  # Change the data to binary form; [0,0,0,0,0,1,0,0,0,0] = 5
    for v in validation:
        binary = [0] * 10
        binary[v] = 1
        set.append(binary)
    return set

################################
#           MAINLINE           #
################################

# Data in original form for testing purposes; Not currently used
# fileIn = readOrigFile("testing-original.txt")
# testingData = fileIn[0]
# testingValidation = fileIn[1]
# fileIn = readOrigFile("training-original.txt")
# trainingData = fileIn[0]
# trainingValidation = fileIn[1]

# Data in preprocessed form
fileIn = readModifiedFile("testing.txt")
testingData = fileIn[0]
testingValidation = fileIn[1]
fileIn = readModifiedFile("training.txt")
trainingData = fileIn[0]
trainingValidation = fileIn[1]
processedTraining = preProcess(trainingValidation)
processedTesting = preProcess(testingValidation)


# Create the ANN
# (learningRate, momentum, maxItr, minMSE, inputNum, hiddenNum, outputNum)
# network = Network(0.05, 0.2, 200, 0.05, 64, 20, 10)
network = Network(0.5, 0.1, 200, 0.05, 64, 20, 10)

user = input("For a full test press enter, for a short test enter S: ")
print("\n\n")
print("Begin Training\n\n")

# A subset of the data is used for ease of testing and marking as the full data set takes a long time to complete
if user == "":
    network.train(trainingData, processedTraining)
    network.test(testingData, processedTesting)
    network.classify(testingData)
else:
    network.train(trainingData[:100], processedTraining[:100])
    network.test(testingData[:50], processedTesting[:50])
    network.classify(testingData[:50])


input("\nPress Enter to Quit")  # Hold the console window open until user closes it
