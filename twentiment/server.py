"""
A ZeroMQ-based server that answeres guessing queries.

:author: 2012, Pascal Hartig
:license: Apache 2
"""

import zmq


class Server:
    def __init__(self, bind="tcp://127.0.0.1:10001"):
        """Creates a new server instance.

        :param bind: The zmq bind, defaults to tcp://127.0.0.1:10001.
            Obviously, the same must be used on the client side.
        """

        self.bind = bind

    def run(self):
        """Starts a blocking server."""

        context = zmq.Context()
        socket = context.socket(zmq.REP)

        socket.bind(self.bind)

        while True:
            message = socket.recv()
            response = self._handle_message(str(message, "utf-8"))
            socket.send_unicode(response)

    def _handle_message(self, message):
        cmd = message.lsplit(" ", 1).lower()

        if cmd == 'guess':
            return self._guess(message)
        else:
            return self._error_response("UNKNOWN_COMMAND")

    def _error_response(self, message):
        return "ERROR {}".format(message)

    def _guess(self, message):
        return "0.0"
