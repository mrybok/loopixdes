import numpy as np

EPS = 1e-13
NUM = (int, float)
REF_TIMESTAMP = 345600
IDX = np.array([1, 3, 7, 11, 15, 19])

BAR_FORMAT = "{percentage:.1f}%|{bar}| {n:.1f}/{total} " \
             "[{elapsed}<{remaining} {postfix}]"

DEFAULT_PARAMS = {
    'DROP': 5.0,
    'LOOP': 5.0,
    'DELAY': 1.0,
    'PAYLOAD': 5.0,
    'LOOP_MIX': 5.0,
}

LABELS = [
    'bandwidth',
    'e2e_entropy',
    'payload_latency',
    'loop_mix_latency',
    'incremental_entropy_mix',
    'incremental_entropy_provider'
]

LOG_STR =                                                                       \
    '                        time: |{:^39s}|                                \n' \
    '               bandwidth use: |{:^39s}|                                \n' \
    '                              -----------------------------------------\n' \
    '                              |   min   |  mean   |   max   |   std   |\n' \
    '                              -----------------------------------------\n' \
    '               e2e anonymity: |{:^9.5f}|{:^9.5f}|{:^9.5f}|{:^9.5f}|    \n' \
    '             payload latency: |{:^9.3f}|{:^9.3f}|{:^9.3f}|{:^9.3f}|    \n' \
    '            loop mix latency: |{:^9.3f}|{:^9.3f}|{:^9.3f}|{:^9.3f}|    \n' \
    '     incremental entropy mix: |{:^9.3f}|{:^9.3f}|{:^9.3f}|{:^9.3f}|    \n' \
    'incremental entropy provider: |{:^9.3f}|{:^9.3f}|{:^9.3f}|{:^9.3f}|      '

# TODO ADD USERS
# '                online users: |{:^39f}|                                \n' \


def default_client_model(timestamp: float) -> int:
    return 100


# TODO CLEAN KWARGS
def default_encryption_model(**kwargs) -> float:
    num_layers = kwargs['_Simulator__num_layers']
    plaintext_size = kwargs['_Simulator__plaintext_size']

    loc = 1.26672150e-7 * plaintext_size + 8.49356836e-4 * num_layers
    loc += 4.07480411e-4
    scale = 1.3898e-4 * num_layers + 4.8268e-4

    return max(kwargs['_Simulator__rng'].normal(loc, scale), EPS)


def default_decryption_model(**kwargs) -> float:
    plaintext_size = kwargs['_Simulator__plaintext_size']

    loc = 1.20195552e-8 * plaintext_size + 8.03016416e-4
    scale = 3.6622e-4

    return max(kwargs['_Simulator__rng'].normal(loc, scale), EPS)


def default_udp_model(**kwargs) -> float:
    rtt = 0.02
    bandwidth_per_thread = 1.5625e7

    num_layers = kwargs['_Simulator__num_layers']
    payload_size = kwargs['_Simulator__plaintext_size']

    packet_size = header_size(num_layers) + payload_size

    return packet_size / bandwidth_per_thread + rtt


def header_size(num_layers: int) -> float:
    return num_layers * 49 + 116
