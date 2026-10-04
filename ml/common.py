COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
"num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
"num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
"dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
CAT = ["protocol_type", "service", "flag"]
NUM = [c for c in COLS if c not in CAT]
GROUPS = {
 "DoS": "back land neptune pod smurf teardrop mailbomb apache2 processtable udpstorm worm".split(),
 "Probe": "ipsweep nmap portsweep satan mscan saint".split(),
 "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named snmpgetattack snmpguess xlock xsnoop httptunnel".split(),
 "U2R": "buffer_overflow loadmodule perl rootkit ps sqlattack xterm".split(),
}
LABEL_MAP = {a: g for g, items in GROUPS.items() for a in items}
LABEL_MAP["normal"] = "Normal"
SEVERITY = {"Normal": "none", "Probe": "medium", "R2L": "high", "DoS": "high", "U2R": "critical"}

def load(path):
    import pandas as pd
    df = pd.read_csv(path, header=None, names=COLS + ["label", "difficulty"])
    df["category"] = df["label"].map(LABEL_MAP)
    if df["category"].isna().any():
        raise ValueError("Unmapped labels: %s" % df.loc[df.category.isna(), "label"].unique())
    return df
