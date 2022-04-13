import numpy as np

EPS = 1e-13
NUM = (int, float)
REF_TIMESTAMP = 345600
IDX = np.array([2, 3, 5, 9, 13, 17])

BAR_FORMAT = "{percentage:.1f}%|{bar}| {n:.1f}/{total:.1f} " \
             "[{elapsed}<{remaining} {postfix}]"

DEFAULT_PARAMS = {
    'DROP': 200/3,
    'LOOP': 200/3,
    'DELAY': 1.0,
    'PAYLOAD': 200/3,
    'LOOP_MIX': 18.0,
}

LABELS = [
    'payload_latency',
    'bandwidth',
    'e2e_entropy',
    'loop_mix_latency',
    'incremental_entropy_mix',
    'incremental_entropy_provider'
]

LOG_STR =                                                                       \
    '                        time: |{:^39s}|                                \n' \
    '                online users: |{:^39d}|                                \n' \
    '             payload latency: |{:^39.5f}|                              \n' \
    '               bandwidth use: |{:^39s}|                                \n' \
    '                              -----------------------------------------\n' \
    '                              |   min   |  mean   |   max   |   std   |\n' \
    '                              -----------------------------------------\n' \
    '               e2e anonymity: |{:^9.5f}|{:^9.5f}|{:^9.5f}|{:^9.5f}|    \n' \
    '            loop mix latency: |{:^9.3f}|{:^9.3f}|{:^9.3f}|{:^9.3f}|    \n' \
    '     incremental entropy mix: |{:^9.3f}|{:^9.3f}|{:^9.3f}|{:^9.3f}|    \n' \
    'incremental entropy provider: |{:^9.3f}|{:^9.3f}|{:^9.3f}|{:^9.3f}|      '


def default_client_model(timestamp: float) -> int:
    return 100


def default_encryption_model(**kwargs) -> float:
    num_layers = kwargs['num_layers']
    plaintext_size = kwargs['plaintext_size']

    loc = 1e-7 * plaintext_size + 8e-4 * num_layers
    loc += 4e-4
    scale = 1e-4 * num_layers + 5e-4

    return max(kwargs['rng'].normal(loc, scale), EPS)


def default_decryption_model(**kwargs) -> float:
    plaintext_size = kwargs['plaintext_size']

    loc = 1e-8 * plaintext_size + 8e-4
    scale = 4e-4

    return max(kwargs['rng'].normal(loc, scale), EPS)


def default_transport_model(**kwargs) -> float:
    propagation_delay = 0.001
    bandwidth_per_thread = 1.5625e7

    num_layers = kwargs['num_layers']
    payload_size = kwargs['plaintext_size']

    packet_size = default_header_size(num_layers) + payload_size

    return packet_size / bandwidth_per_thread + propagation_delay


def default_header_size(num_layers: int) -> int:
    return num_layers * 49 + 116
