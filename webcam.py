from threading import Thread
import cv2


# Allows for significant reduction in latency when reading from camera
class WebcamVideoStream:
    def __init__(self, src=0):
        # Initialize camera stream and read first frame
        self.stream = cv2.VideoCapture(src)
        _, self.frame = self.stream.read()

        # Keep track of whether thread should be stopped
        self.stopped = False

    def start(self):
        # Initialize and start thread responsible for reading frames from camera stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Continuously read frames until stopped is True
        while self.stream.isOpened():
            if self.stopped:
                return

            _, self.frame = self.stream.read()

    def read(self):
        # Get latest frame
        return self.frame

    def stop(self):
        print("Stopping stream")

        # Indicate thread should be stopped
        self.stopped = True
