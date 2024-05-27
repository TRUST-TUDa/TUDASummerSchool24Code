import struct
import torch
import numpy
import json
from collections import OrderedDict
from TUDASummerSchool24Code.Utils import print_timed

"Convert all tensors in the dictionary to float lists for json"
def tensor_to_float(model):
    serializable_model = OrderedDict()
    for key, value in model.items():
        serializable_model[key] = value.cpu().numpy().tolist()
    return serializable_model

"Convert all float lists back to tensors"
def float_to_tensor(model):
    tensor_model = OrderedDict()
    for key, value in model.items():
        tensor_model[key] = torch.tensor(numpy.asarray(value))
    return tensor_model

"Helper function to recv n bytes or return None if EOF is hit"
def recv_n(sock, n, verbose=True):
    data = bytearray()
    if verbose:
        print_timed(f"Receiving message of length {n}...")
    next_print_boundary = n/10
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

"Read message length, then receive a message of that length, decode and dejson"
def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recv_n(sock, 4, verbose=False)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]

    # Read the message data
    return json.loads(recv_n(sock, msglen).decode('utf-8'))

"jsonify message, calculate length and send it over the socket"
def send_msg(sock, jsonable):
    msg = json.dumps(jsonable, sort_keys=True).encode()
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

