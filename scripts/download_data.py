#!/usr/bin/env python3
"""Download security datasets for training.

Supports:
- UNSW-NB15: Network intrusion detection dataset
- CTU-13: Botnet traffic dataset
- Synthetic: Generate synthetic security events
"""

import argparse
import os
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import gzip
import shutil

DATA_DIR = Path(__file__).parent.parent / "data"

DATASETS = {
    "unsw-nb15": {
        "description": "UNSW-NB15 Network Intrusion Dataset",
        "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "files": {
            # These are the actual download URLs from UNSW
            "training": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download",
            "testing": "https://cloudstor.aarnet.edu.au/plus/s/fG78L5e0JlfE9Co/download",
        },
        "size": "~150MB",
    },
    "ctu-13": {
        "description": "CTU-13 Botnet Traffic Dataset",
        "url": "https://www.stratosphereips.org/datasets-ctu13",
        "note": "Manual download required due to size",
    },
    "synthetic": {
        "description": "Synthetic security events for testing",
        "generator": True,
    },
}


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def generate_synthetic_data(output_dir: Path, n_events: int = 100000):
    """Generate synthetic security events for testing."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    print(f"Generating {n_events} synthetic security events...")
    
    np.random.seed(42)
    
    # Base timestamp
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # Generate normal traffic (90%)
    n_normal = int(n_events * 0.9)
    n_attack = n_events - n_normal
    
    # Normal traffic patterns
    normal_events = []
    for i in range(n_normal):
        event = {
            "timestamp": start_time + timedelta(seconds=i * 0.5 + np.random.random()),
            "src_ip": f"192.168.1.{np.random.randint(1, 255)}",
            "dst_ip": f"10.0.0.{np.random.randint(1, 50)}",
            "src_port": np.random.randint(1024, 65535),
            "dst_port": np.random.choice([80, 443, 22, 53, 25, 3306]),
            "protocol": np.random.choice(["http", "https", "ssh", "dns", "smtp"]),
            "bytes_in": np.random.exponential(500),
            "bytes_out": np.random.exponential(200),
            "duration": np.random.exponential(0.5),
            "event_type": "connection",
            "status": "success",
            "label": 0,
            "attack_category": "normal",
        }
        normal_events.append(event)
    
    # Attack traffic patterns
    attack_types = [
        ("port_scan", 0.3),
        ("brute_force", 0.2),
        ("data_exfil", 0.15),
        ("dos", 0.2),
        ("probe", 0.15),
    ]
    
    attack_events = []
    for i in range(n_attack):
        attack_type = np.random.choice(
            [a[0] for a in attack_types],
            p=[a[1] for a in attack_types]
        )
        
        base_time = start_time + timedelta(seconds=np.random.randint(0, n_normal // 2))
        
        if attack_type == "port_scan":
            event = {
                "timestamp": base_time + timedelta(seconds=i * 0.01),
                "src_ip": f"192.168.1.{np.random.randint(1, 10)}",
                "dst_ip": f"10.0.0.{np.random.randint(1, 5)}",
                "src_port": np.random.randint(1024, 65535),
                "dst_port": np.random.randint(1, 65535),  # Random ports (scan)
                "protocol": "tcp",
                "bytes_in": 0,
                "bytes_out": np.random.randint(40, 60),
                "duration": 0.01,
                "event_type": "connection",
                "status": np.random.choice(["failed", "success"], p=[0.8, 0.2]),
                "label": 1,
                "attack_category": "port_scan",
            }
        elif attack_type == "brute_force":
            event = {
                "timestamp": base_time + timedelta(seconds=i * 0.1),
                "src_ip": f"192.168.1.{np.random.randint(200, 255)}",
                "dst_ip": "10.0.0.1",
                "src_port": np.random.randint(1024, 65535),
                "dst_port": 22,
                "protocol": "ssh",
                "bytes_in": np.random.randint(100, 200),
                "bytes_out": np.random.randint(50, 100),
                "duration": np.random.exponential(0.1),
                "event_type": "auth",
                "status": "failed",
                "label": 1,
                "attack_category": "brute_force",
            }
        elif attack_type == "data_exfil":
            event = {
                "timestamp": base_time + timedelta(seconds=i),
                "src_ip": "192.168.1.50",
                "dst_ip": f"8.8.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                "src_port": np.random.randint(1024, 65535),
                "dst_port": 443,
                "protocol": "https",
                "bytes_in": np.random.randint(100, 500),
                "bytes_out": np.random.randint(100000, 10000000),  # Large outbound
                "duration": np.random.exponential(5),
                "event_type": "connection",
                "status": "success",
                "label": 1,
                "attack_category": "data_exfil",
            }
        elif attack_type == "dos":
            event = {
                "timestamp": base_time + timedelta(seconds=i * 0.001),  # Very fast
                "src_ip": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                "dst_ip": "10.0.0.1",
                "src_port": np.random.randint(1024, 65535),
                "dst_port": 80,
                "protocol": "http",
                "bytes_in": 0,
                "bytes_out": np.random.randint(1000, 2000),
                "duration": 0.001,
                "event_type": "connection",
                "status": "failed",
                "label": 1,
                "attack_category": "dos",
            }
        else:  # probe
            event = {
                "timestamp": base_time + timedelta(seconds=i * 0.5),
                "src_ip": f"192.168.1.{np.random.randint(1, 10)}",
                "dst_ip": f"10.0.0.{np.random.randint(1, 255)}",  # Many targets
                "src_port": np.random.randint(1024, 65535),
                "dst_port": np.random.choice([80, 443, 22, 3389]),
                "protocol": np.random.choice(["http", "ssh", "rdp"]),
                "bytes_in": np.random.randint(0, 100),
                "bytes_out": np.random.randint(40, 100),
                "duration": np.random.exponential(0.1),
                "event_type": "connection",
                "status": np.random.choice(["failed", "success"], p=[0.6, 0.4]),
                "label": 1,
                "attack_category": "probe",
            }
        
        attack_events.append(event)
    
    # Combine and sort by timestamp
    all_events = normal_events + attack_events
    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_events.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} events ({n_normal} normal, {n_attack} attacks)")
    print(f"Saved to: {output_path}")
    
    # Print attack distribution
    print("\nAttack distribution:")
    print(df[df["label"] == 1]["attack_category"].value_counts())
    
    return output_path


def download_unsw_nb15(output_dir: Path):
    """Download UNSW-NB15 dataset."""
    print("UNSW-NB15 Dataset")
    print("=" * 50)
    print("Due to licensing, please download manually from:")
    print("https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print()
    print("Download the following files:")
    print("  - UNSW-NB15_1.csv to UNSW-NB15_4.csv (training data)")
    print("  - UNSW_NB15_testing-set.csv (testing data)")
    print()
    print(f"Place files in: {output_dir / 'unsw-nb15'}")
    
    # Create directory
    (output_dir / "unsw-nb15").mkdir(parents=True, exist_ok=True)
    
    # Create a sample features file showing expected columns
    sample = """# Expected columns in UNSW-NB15:
srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,sloss,dloss,service,Sload,Dload,Spkts,Dpkts,swin,dwin,stcpb,dtcpb,smeansz,dmeansz,trans_depth,res_bdy_len,Sjit,Djit,Stime,Ltime,Sintpkt,Dintpkt,tcprtt,synack,ackdat,is_sm_ips_ports,ct_state_ttl,ct_flw_http_mthd,is_ftp_login,ct_ftp_cmd,ct_srv_src,ct_srv_dst,ct_dst_ltm,ct_src_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,attack_cat,Label
"""
    
    with open(output_dir / "unsw-nb15" / "COLUMNS_INFO.txt", "w") as f:
        f.write(sample)


def main():
    parser = argparse.ArgumentParser(description="Download security datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="synthetic",
        help="Dataset to download/generate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=100000,
        help="Number of events for synthetic dataset"
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "synthetic":
        generate_synthetic_data(args.output_dir / "synthetic", args.n_events)
    elif args.dataset == "unsw-nb15":
        download_unsw_nb15(args.output_dir)
    else:
        print(f"Dataset '{args.dataset}' requires manual download.")
        info = DATASETS[args.dataset]
        print(f"Description: {info.get('description', 'N/A')}")
        print(f"URL: {info.get('url', 'N/A')}")


if __name__ == "__main__":
    main()
