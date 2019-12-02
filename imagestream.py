from threading import Thread
import socket
import queue
import cv2
import numpy as np

class ImageStreamThread(Thread):
    def __init__(self):
        super(ImageStreamThread, self).__init__()
        # Initialize
        self.socket = socket.socket()
        print("Socket created")
        self.socket.bind(("", 9002))
        self.socket.listen(5)
        print("Socket binded to port 9002 and listening")

        # Waiting for client socket socket to connect
        self.conn, addr = self.socket.accept()
        print(f"Connected by: {addr}")

        self.frame = None
        self.stopped = False
        self.result = None

    #def start(self):
    #    # Initialize and start thread responsible for reading image bytes from socket
    #    Thread(target=self.receive, args=()).start()
    #    Thread(target=self.send, args=()).start()
    #    return self

    def run(self):
        while not self.stopped:
            if self.result is not None:
                self.conn.send(self.result.encode('utf-8'))
                self.result = None

            print("About to receive image size")
            # Receive size from client and convert to integer
            image_size = int.from_bytes(self.conn.recv(4), byteorder="big")
            print(f"\nReceived size: {image_size}")

            # Stop if connection was closed on client side
            if image_size == 0:
                print("Connection closed, stopping")
                self.stop()
                break

            # Send back size that was received to confirm that it is correct
            image_size_response = "SIZE\n" + str(image_size) + "\n"
            print(f"Image size response: {image_size_response}");
            self.conn.send(image_size_response.encode('utf-8'))

            # Keep reading from connection until enough bytes have been read to reach the image size
            total_data = []
            while len(total_data) < image_size:
                data = self.conn.recv(1024)

                # Stop if connection was closed on client side
                if data == 0:
                    print("Connection closed, stopping")
                    self.stop()
                    break

                total_data.extend(data)

            print("Received image bytes")

            # Send confirmation that image was received back to client
            ok = "OK\n"
            self.conn.send(ok.encode('utf-8'))

            # Convert bytes to bytearray, then to numpy array, then to cv2 matrix for image
            total_data = bytearray(total_data)
            total_data = np.asarray(total_data)
            self.frame = cv2.imdecode(total_data, cv2.IMREAD_COLOR)
            print("Decoded frame from bytes, printed below:")
            print(self.frame)


    #def receive(self):
        #self.conn.send("DONE\n".encode('utf-8'))
        #self.socket.close()


    def read_frame(self):
        while self.frame is None:
            pass
        # Get latest frame
        return self.frame


    #def send(self):
        #print("Sending results")
        #self.conn.send(result.encode())
        #self.result = result
        #self.send_ready = True
    #    while not self.stopped:
    #        data = self.data_queue.get()
    #        if type(data) is int:
    #            self.conn.send(data.to_bytes(4, byteorder="big"))
    #        elif type(data) is str:
    #            self.conn.send(data.encode('utf-8'))
    #        else:
    #            print("Don't recognize data type, not sending")


    def send_result(self, result):
        self.result = "RESULT\n" + result

    #def get_socket(self):
    #    return self.socket

    def stop(self):
        print("-> Stopping stream")

        # Indicate thread should be stopped
        self.stopped = True
        self.socket.close()
        print("Called close on socket")
