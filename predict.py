# imports
import cv2 as cv
import numpy as np
import joblib as jl
from skimage.feature import hog
from skimage.feature import local_binary_pattern as lbp
# imports


class predict:
    def __init__(self):
        pass

    def resizeImg(self, image):
        return cv.resize(image, (150, 150))

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


def main():
    # opening webcam
    cam = cv.VideoCapture(0)
    pr = predict()
    # loading model that trained and tested with 83.5% accuracy
    Model = jl.load("Model")
    while True:
        X = []
        # capture frame of video
        check, fr = cam.read()
        # extract face from frame
        croppedFr = pr.cropImage(fr)
        # resize cropped frame
        resizedCroppedFr = pr.resizeImg(croppedFr)
        # append lbp feature adn hog feature using np.append
        X.append(np.append(pr.HogMethod(resizedCroppedFr),
                 pr.LbpMethod(resizedCroppedFr)))
        # predict smile from frame if predictedResult == 1 then print smile else print no smile
        predictedResult = Model.predict(X)
        fr = cv.putText(
            fr,
            "Smile" if predictedResult == ["1"] else "No Smile",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if predictedResult == ["1"] else (0, 0, 255),
            2,
        )
        cv.imshow("Webcam", fr)
        # click esc to exit from smile detector
        key = cv.waitKey(1)
        if key == 27:
            break
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
