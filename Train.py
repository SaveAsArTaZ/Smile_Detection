# imports
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
import joblib as jl
from skimage.feature import hog
from skimage.feature import local_binary_pattern as lbp
from sklearn.model_selection import train_test_split as tts
# imports


class Train_And_Test:
    # constructor function with variables TrainX,TrainY,TestX,TestY,X,Y,Path,Model
    def __init__(self):
        (self.TrainX,
         self.TrainY,
         self.TestX,
         self.TestY,
         self.X,
         self.Y,
         self.Path,
         self.model) = ([], [], [], [], [], [], "genki4k/files/file", None)

    # crop image function
    def cropImage(self, image):
        # convert to grayscale of each frames
        grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # read the haarcascade to detect the faces in an image
        faceCascade = cv.CascadeClassifier(
            'haarcascade_frontalface_alt.xml')

        # detects faces in the input image
        faces = faceCascade.detectMultiScale(grayImg, 1.25, 5)
        # loop over all detected faces
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # To draw a rectangle in a face
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
                Face = image[y:y+h, x:x+w]
                return Face
        else:
            return image

    # apply hog feature method
    def HogMethod(self, image):
        fd, hg = hog(image,
                     orientations=20,
                     pixels_per_cell=(20, 20),
                     cells_per_block=(1, 1),
                     visualize=True,
                     channel_axis=-1)
        return fd

    # apply lbp feature method
    def LbpMethod(self, image):
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        localBinaryPattern = lbp(grayImage, 3 * 9, 3, method="uniform")
        (h, _) = np.histogram(localBinaryPattern.ravel(),
                              bins=np.arange(24), range=(0, 29))
        h = h.astype("float")
        h /= (h.sum() + 1e-7)
        return h

    # calculate X array(result of hog and lbp features) method
    def calc_X(self):
        for i in range(1, 4001):
            # read image with regex genki4k/files/file + XXXX + .jpg
            image = cv.imread(self.Path + str("%04d" % i) + ".jpg")
            croppedImage = self.cropImage(image)
            resizedCroppedImage = cv.resize(croppedImage, (150, 150))
            # append two features together to calculate X
            self.X.append(np.append(
                self.HogMethod(resizedCroppedImage),
                self.LbpMethod(resizedCroppedImage)))

    # calculate Y array(labels) method
    def calc_Y(self):
        my_file = open("genki4k/labels.txt", "r")
        data = my_file.read()
        datalist = data.split("\n")
        for i in range(1, 4001):
            tmp = datalist[i].split(' ')
            # just the first character of each line needed
            self.Y.append(tmp[0])
        my_file.close()

    # calculate x and y and split it into TrainX , TrainY , TestX , TestY
    def splitTestTrain(self):
        self.calc_X()
        self.calc_Y()
        self.TrainX, self.TestX, self.TrainY, self.TestY = tts(self.X, self.Y,
                                                               random_state=104,
                                                               train_size=0.7, shuffle=True)

    # main function to train our algorithm and test its accuracy
    def trainTestMethod(self):
        # calculate X , Y and split it into TrainX , TrainY , TestX , TestY
        self.splitTestTrain()
        # init value of MAX SCORE AND MAX C
        maxResult, maxC = -1, -1
        # applying svc in range 0.05 to 2 with seq 0.05 to see the best result
        for i in [float(j) / 100 for j in range(5, 205, 5)]:
            m = SVC(kernel="rbf", C=i)
            m.fit(self.TrainX, self.TrainY)
            s = m.score(self.TestX, self.TestY)
            if maxResult == -1:
                maxResult = s
                maxC = i
            elif maxResult < s:
                maxResult = s
                maxC = i
        print("accuracy is : " + maxResult)
        # create the best model and saving it
        self.model = SVC(kernel="rbf", C=maxC)
        self.model.fit(self.TrainX, self.TrainY)
        jl.dump(self.model, "Model")


def main():
    tat = Train_And_Test()
    tat.trainTestMethod()


if __name__ == "__main__":
    main()
