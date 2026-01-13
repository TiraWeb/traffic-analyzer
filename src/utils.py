import random
import streamlit as st

# KDD feature names (kept in same order expected by the models)
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate'
]

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
PROTOCOL_OPTIONS = ['tcp', 'udp', 'icmp']
SERVICE_OPTIONS = [
    'http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i',
    'ntp_u', 'ecr_i', 'other'
]
FLAG_OPTIONS = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH']


def generate_test_data() -> dict:
    """Create a semi-realistic KDD-style feature dict for demo/live generation.

    This mirrors previous behavior in the monolithic `main.py`. It returns a dict
    of FEATURE_NAMES -> values. Uses `st.info` to notify UI about the generated
    pattern (attack vs normal).
    """
    d = {}
    is_attack = random.random() < 0.5

    if is_attack:
        attack_patterns = [
            {
                'name': 'Smurf/Neptune DoS',
                'duration': 0,
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 511,
                'srv_count': 511,
                'serror_rate': 1.0,
                'srv_serror_rate': 1.0,
                'same_srv_rate': 1.0,
                'dst_host_count': 255,
                'dst_host_srv_count': 255,
                'protocol_type': 'icmp',
                'service': 'eco_i',
                'flag': 'SF'
            },
            {
                'name': 'Satan/Nmap probe',
                'duration': 0,
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 150,
                'srv_count': 25,
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'diff_srv_rate': 1.0,
                'dst_host_count': 255,
                'dst_host_same_srv_rate': 0.0,
                'dst_host_diff_srv_rate': 1.0,
                'protocol_type': 'tcp',
                'service': 'other',
                'flag': 'S0'
            },
            {
                'name': 'Buffer overflow',
                'num_compromised': 100,
                'root_shell': 1,
                'su_attempted': 2,
                'hot': 30,
                'count': 1,
                'srv_count': 1,
                'src_bytes': 1000,
                'dst_bytes': 0,
                'protocol_type': 'tcp',
                'service': 'telnet',
                'flag': 'SF'
            },
            {
                'name': 'FTP write',
                'num_file_creations': 10,
                'num_access_files': 10,
                'logged_in': 1,
                'num_compromised': 5,
                'hot': 15,
                'protocol_type': 'tcp',
                'service': 'ftp',
                'flag': 'SF'
            }
        ]

        pattern = random.choice(attack_patterns)
        st.info(f"Generated ATTACK pattern: {pattern.get('name', 'attack')}")

        d['protocol_type'] = pattern.get('protocol_type', random.choice(PROTOCOL_OPTIONS))
        d['service'] = pattern.get('service', random.choice(SERVICE_OPTIONS))
        d['flag'] = pattern.get('flag', random.choice(FLAG_OPTIONS))

        d['duration'] = pattern.get('duration', random.randint(0, 1000))
        d['src_bytes'] = pattern.get('src_bytes', random.randint(0, 20000))
        d['dst_bytes'] = pattern.get('dst_bytes', random.randint(0, 20000))
        d['land'] = pattern.get('land', random.choice([0, 1]))
        d['wrong_fragment'] = pattern.get('wrong_fragment', random.randint(0, 5))
        d['urgent'] = pattern.get('urgent', random.randint(0, 3))
        d['hot'] = pattern.get('hot', random.randint(0, 10))
        d['num_failed_logins'] = pattern.get('num_failed_logins', random.randint(0, 5))
        d['logged_in'] = pattern.get('logged_in', random.choice([0, 1]))
        d['num_compromised'] = pattern.get('num_compromised', random.randint(0, 100))
        d['root_shell'] = pattern.get('root_shell', random.choice([0, 1]))
        d['su_attempted'] = pattern.get('su_attempted', random.choice([0, 1]))
        d['num_root'] = pattern.get('num_root', random.randint(0, 10))
        d['num_file_creations'] = pattern.get('num_file_creations', random.randint(0, 20))
        d['num_shells'] = pattern.get('num_shells', random.randint(0, 5))
        d['num_access_files'] = pattern.get('num_access_files', random.randint(0, 10))
        d['num_outbound_cmds'] = pattern.get('num_outbound_cmds', random.randint(0, 5))
        d['is_host_login'] = pattern.get('is_host_login', random.choice([0, 1]))
        d['is_guest_login'] = pattern.get('is_guest_login', random.choice([0, 1]))
        d['count'] = pattern.get('count', random.randint(1, 500))
        d['srv_count'] = pattern.get('srv_count', random.randint(1, 500))
        d['serror_rate'] = pattern.get('serror_rate', random.uniform(0, 1))
        d['srv_serror_rate'] = pattern.get('srv_serror_rate', random.uniform(0, 1))
        d['rerror_rate'] = pattern.get('rerror_rate', random.uniform(0, 1))
        d['srv_rerror_rate'] = pattern.get('srv_rerror_rate', random.uniform(0, 1))
        d['same_srv_rate'] = pattern.get('same_srv_rate', random.uniform(0, 1))
        d['diff_srv_rate'] = pattern.get('diff_srv_rate', random.uniform(0, 1))
        d['srv_diff_host_rate'] = pattern.get('srv_diff_host_rate', random.uniform(0, 1))
        d['dst_host_count'] = pattern.get('dst_host_count', random.randint(0, 255))
        d['dst_host_srv_count'] = pattern.get('dst_host_srv_count', random.randint(0, 255))
        d['dst_host_same_srv_rate'] = pattern.get('dst_host_same_srv_rate', random.uniform(0, 1))
        d['dst_host_diff_srv_rate'] = pattern.get('dst_host_diff_srv_rate', random.uniform(0, 1))
        d['dst_host_same_src_port_rate'] = pattern.get('dst_host_same_src_port_rate', random.uniform(0, 1))
        d['dst_host_srv_diff_host_rate'] = pattern.get('dst_host_srv_diff_host_rate', random.uniform(0, 1))
    else:
        st.info("Generated NORMAL pattern")
        d['protocol_type'] = random.choice(PROTOCOL_OPTIONS)
        d['service'] = random.choice(SERVICE_OPTIONS)
        d['flag'] = random.choice(['SF', 'S1'])
        d['duration'] = random.randint(0, 1000)
        d['src_bytes'] = random.randint(0, 20000)
        d['dst_bytes'] = random.randint(0, 20000)
        d['land'] = 0
        d['wrong_fragment'] = random.randint(0, 2)
        d['urgent'] = random.randint(0, 1)
        d['hot'] = random.randint(0, 3)
        d['num_failed_logins'] = 0
        d['logged_in'] = 1
        d['num_compromised'] = 0
        d['root_shell'] = 0
        d['su_attempted'] = 0
        d['num_root'] = random.randint(0, 2)
        d['num_file_creations'] = random.randint(0, 5)
        d['num_shells'] = random.randint(0, 2)
        d['num_access_files'] = random.randint(0, 3)
        d['num_outbound_cmds'] = 0
        d['is_host_login'] = random.choice([0, 1])
        d['is_guest_login'] = 0
        d['count'] = random.randint(1, 50)
        d['srv_count'] = random.randint(1, 50)
        d['serror_rate'] = random.uniform(0, 0.2)
        d['srv_serror_rate'] = random.uniform(0, 0.2)
        d['rerror_rate'] = random.uniform(0, 0.2)
        d['srv_rerror_rate'] = random.uniform(0, 0.2)
        d['same_srv_rate'] = random.uniform(0.5, 1.0)
        d['diff_srv_rate'] = random.uniform(0, 0.3)
        d['srv_diff_host_rate'] = random.uniform(0, 0.3)
        d['dst_host_count'] = random.randint(0, 50)
        d['dst_host_srv_count'] = random.randint(0, 50)
        d['dst_host_same_srv_rate'] = random.uniform(0.5, 1.0)
        d['dst_host_diff_srv_rate'] = random.uniform(0, 0.3)
        d['dst_host_same_src_port_rate'] = random.uniform(0.5, 1.0)
        d['dst_host_srv_diff_host_rate'] = random.uniform(0, 0.3)

    return d
