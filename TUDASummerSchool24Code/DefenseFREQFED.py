import socket
import numpy as np
from TUDASummerSchool24Code.NetworkUtils import *
from TUDASummerSchool24Code.Utils import print_timed

SERVERS = [('130.83.76.1', 4242), ('130.83.76.1', 4243), ('130.83.76.1', 4244), ('130.83.76.1', 4245), ('130.83.76.1', 4246), ('130.83.76.1', 4247)]

class FREQFED:

    def __init__(self, server_host=None, server_port=None):
        self.description = 'FREQFED'
        self.host = server_host
        self.port = server_port

    def __call__(self, global_model_state_dict, all_models, number_of_benign_clients, number_of_malicious_clients):

        print_timed('Serialize Models')
        serializable_model = tensor_to_float(global_model_state_dict)

        serializable_all = []
        for model in all_models:
            serializable_all.append(tensor_to_float(model))

        message = OrderedDict()
        message["global_model"] = serializable_model
        message["all_models"] = serializable_all
        message["number_of_benign_clients"] = number_of_benign_clients
        message["number_of_malicious_clients"] = number_of_malicious_clients

        #print_timed(f'Connect to Server {self.host}')
        # Create a socket (SOCK_STREAM means a TCP socket)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.host is None:
          host, port = SERVERS[np.random.randint(len(SERVERS))]
        else:
          host, port = self.host, self.port
        try:
            sock.connect((host, port))
            print_timed('Send Request')
            send_msg(sock, message)
            print_timed('Receive Answer')
            received = recv_msg(sock)
            print_timed('Deserialize Answer')
            return float_to_tensor(received)
        except:
            print_timed('An Error occurred')
        finally:
            sock.close()
