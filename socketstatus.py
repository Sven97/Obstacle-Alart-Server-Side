from threading import Thread
import socket

class SocketStatusThread(Thread):
    def __init__(self):
        super(SocketStatusThread, self).__init__()
        self.socket = socket.socket()
        print("Status socket created")
        self.socket.bind(("",9003))
        self.socket.listen(5)
        print("Status socket binded to port 9003 and listening")

        self.conn, addr = self.socket.accept()
        print(f"Status socket connected by: {addr}")

        ready_confirmation = "READY\n"
        self.conn.send(ready_confirmation.encode("utf-8"))

    def run(self):
        print("Reading from client")
        while(True):
            self.conn.recv(5)
