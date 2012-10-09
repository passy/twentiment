"""
A ZeroMQ-based server that answeres guessing queries.

:author: 2012, Pascal Hartig
:license: Apache 2
"""

import zmq
from twentiment.extract import extract_features
from twentiment.text import normalize_text


class Server:
    """A simple ZMQ-based server listening on a customizable bind.

    Protocol*::

        -> GUESS [tweet:str]
        <- OK [guess:float]
        - OR -
        <- ERROR [code:str] [description?:str]

    Possible Errors:

        * UNKNOWN_COMMAND: The leading command was not understood.
        * RUNTIME_ERROR: An error on the server side occured.
        * BAD_FORMAT: A request must start with a command separated by an
            ASCII space (20)

    (* Not really worth calling it that.)
    """
    def __init__(self, classifier, bind="tcp://127.0.0.1:10001"):
        """Creates a new server instance.

        :param bind: The zmq bind, defaults to tcp://127.0.0.1:10001.
            Obviously, the same must be used on the client side.
        """

        self.bind = bind
        self.classifier = classifier

    def run(self):
        """Starts a blocking server."""

        context = zmq.Context()
        socket = context.socket(zmq.REP)

        socket.bind(self.bind)

        print("Starting server on {}".format(self.bind))
        while True:
            message = socket.recv()
            try:
                response = self._handle_message(str(message, "utf-8"))
            except Exception as err:
                response = self._error_response("RUNTIME_ERROR " + str(err))
                socket.send_unicode(response)

                raise

            socket.send_unicode(response)

    def _handle_message(self, message):
        try:
            cmd, message = message.lower().split(" ", 1)
        except ValueError:
            return self._error_response("BAD_FORMAT")

        if cmd == 'guess':
            return "OK {}".format(self._guess(message))
        else:
            return self._error_response("UNKNOWN_COMMAND")

    def _error_response(self, message):
        return "ERROR {}".format(message)

    def _guess(self, message):
        twfeat = extract_features(normalize_text(message))
        result = self.classifier.prob_classify(twfeat)
        return format(result.prob('positive') - result.prob('negative'))
