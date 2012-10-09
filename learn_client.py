import zmq


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://*:10001')

    while True:
        line = input("twentiment> ")
        if not line:
            break

        socket.send_unicode(line)
        print(socket.recv())


if __name__ == "__main__":
    main()
