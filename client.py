import socket

class GetLabel(object):
    def __init__(self, host='', port=18763):
        self.HOST = host
        self.PORT = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST, self.PORT))
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.s.send('exit'.encode('utf-8'))
        self.s.close()
    def ask(self):
        self.s.send('label'.encode('utf-8'))
        data = self.s.recv(64)
        return data

if __name__ == '__main__': # testing
    with GetLabel() as askobj:
        for _ in xrange(60):
            print askobj.ask()

