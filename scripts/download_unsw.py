#!/usr/bin/env python3
"""Download and prepare UNSW-NB15 dataset.

The UNSW-NB15 dataset is a modern network intrusion detection dataset
containing 9 types of attacks with 49 features per flow.

Usage:
    python scripts/download_unsw.py
    python scripts/download_unsw.py --sample 50000  # Quick test
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Dataset URLs (hosted on UNSW servers)
URLS = {
    "train": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
    # Alternative: Kaggle mirror (requires kaggle CLI)
    "kaggle": "dhoogla/unswnb15",
}

# Known feature columns in UNSW-NB15
FEATURE_COLS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
]

CATEGORICAL_COLS = ['proto', 'service', 'state']
LABEL_COL = 'label'
ATTACK_CAT_COL = 'attack_cat'


def download_from_kaggle(output_dir: Path):
    """Download UNSW-NB15 from Kaggle (requires kaggle CLI)."""
    import subprocess
    
    console.print("[yellow]Downloading from Kaggle...[/yellow]")
    console.print("Note: Requires 'kaggle' CLI and API key configured")
    
    try:
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "dhoogla/unswnb15",
            "-p", str(output_dir), "--unzip"
        ], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[red]Kaggle download failed: {e}[/red]")
        return False


def create_sample_from_original(output_dir: Path, n_samples: int = 50000):
    """Create a smaller sample dataset for quick testing.
    
    If you have the full dataset, this extracts a stratified sample.
    """
    full_path = output_dir / "UNSW_NB15_training-set.csv"
    
    if not full_path.exists():
        console.print(f"[yellow]Full dataset not found at {full_path}[/yellow]")
        console.print("Creating synthetic UNSW-NB15-like sample for demo...")
        return create_demo_sample(output_dir, n_samples)
    
    console.print(f"[blue]Loading full dataset from {full_path}...[/blue]")
    df = pd.read_csv(full_path)
    
    # Stratified sample
    from sklearn.model_selection import train_test_split
    sample, _ = train_test_split(
        df, 
        train_size=n_samples,
        stratify=df[LABEL_COL],
        random_state=42
    )
    
    sample_path = output_dir / f"unsw_nb15_sample_{n_samples}.csv"
    sample.to_csv(sample_path, index=False)
    console.print(f"[green]✓ Sample saved to {sample_path}[/green]")
    
    return sample_path


def create_demo_sample(output_dir: Path, n_samples: int = 50000):
    """Create a realistic UNSW-NB15-like demo dataset.
    
    This mimics the statistical properties of UNSW-NB15 for demo purposes
    when the real dataset isn't available.
    """
    np.random.seed(42)
    
    # Attack distribution (roughly matches UNSW-NB15)
    attack_types = {
        'Normal': 0.68,
        'Generic': 0.08,
        'Exploits': 0.07,
        'Fuzzers': 0.05,
        'DoS': 0.04,
        'Reconnaissance': 0.03,
        'Analysis': 0.02,
        'Backdoor': 0.02,
        'Shellcode': 0.005,
        'Worms': 0.005,
    }
    
    # Generate labels
    attack_cats = np.random.choice(
        list(attack_types.keys()),
        size=n_samples,
        p=list(attack_types.values())
    )
    labels = (attack_cats != 'Normal').astype(int)
    
    # Generate features with realistic distributions
    data = {
        # Duration (log-normal, attacks often shorter)
        'dur': np.exp(np.random.normal(0, 2, n_samples)) * (1 - 0.5 * labels),
        
        # Packet counts
        'spkts': np.random.poisson(10, n_samples) + 1,
        'dpkts': np.random.poisson(8, n_samples) + 1,
        
        # Bytes (log-normal)
        'sbytes': np.exp(np.random.normal(7, 2, n_samples)).astype(int),
        'dbytes': np.exp(np.random.normal(6, 2, n_samples)).astype(int),
        
        # Rate features
        'rate': np.random.exponential(100, n_samples),
        'sttl': np.random.choice([32, 64, 128, 255], n_samples),
        'dttl': np.random.choice([32, 64, 128, 255], n_samples),
        
        # Load features
        'sload': np.random.exponential(1000, n_samples),
        'dload': np.random.exponential(800, n_samples),
        
        # Loss features (higher for attacks)
        'sloss': np.random.poisson(2, n_samples) * (1 + labels),
        'dloss': np.random.poisson(1, n_samples) * (1 + labels),
        
        # Inter-packet timing
        'sinpkt': np.random.exponential(50, n_samples),
        'dinpkt': np.random.exponential(60, n_samples),
        
        # Jitter
        'sjit': np.abs(np.random.normal(0, 50, n_samples)),
        'djit': np.abs(np.random.normal(0, 40, n_samples)),
        
        # Window sizes
        'swin': np.random.choice([0, 255, 65535], n_samples, p=[0.1, 0.3, 0.6]),
        'dwin': np.random.choice([0, 255, 65535], n_samples, p=[0.1, 0.3, 0.6]),
        
        # TCP features
        'stcpb': np.random.randint(0, 2**32, n_samples),
        'dtcpb': np.random.randint(0, 2**32, n_samples),
        'tcprtt': np.random.exponential(0.1, n_samples),
        'synack': np.random.exponential(0.05, n_samples),
        'ackdat': np.random.exponential(0.05, n_samples),
        
        # Mean packet sizes
        'smean': np.random.lognormal(5, 1, n_samples),
        'dmean': np.random.lognormal(5, 1, n_samples),
        
        # Connection tracking features
        'trans_depth': np.random.poisson(1, n_samples),
        'response_body_len': np.random.exponential(500, n_samples).astype(int),
        'ct_srv_src': np.random.poisson(3, n_samples),
        'ct_state_ttl': np.random.poisson(2, n_samples),
        'ct_dst_ltm': np.random.poisson(2, n_samples),
        'ct_src_dport_ltm': np.random.poisson(2, n_samples),
        'ct_dst_sport_ltm': np.random.poisson(2, n_samples),
        'ct_dst_src_ltm': np.random.poisson(2, n_samples),
        'is_ftp_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'ct_ftp_cmd': np.random.poisson(0.1, n_samples),
        'ct_flw_http_mthd': np.random.poisson(0.5, n_samples),
        'ct_src_ltm': np.random.poisson(3, n_samples),
        'ct_srv_dst': np.random.poisson(2, n_samples),
        'is_sm_ips_ports': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        
        # Categorical
        'proto': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.8, 0.15, 0.05]),
        'service': np.random.choice(['http', 'ftp', 'ssh', 'dns', '-'], n_samples, p=[0.4, 0.1, 0.1, 0.1, 0.3]),
        'state': np.random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST'], n_samples),
        
        # Labels
        'attack_cat': attack_cats,
        'label': labels,
    }
    
    df = pd.DataFrame(data)
    
    # Add some attack-specific patterns
    # DoS: high packet rate, short duration
    dos_mask = df['attack_cat'] == 'DoS'
    df.loc[dos_mask, 'rate'] *= 10
    df.loc[dos_mask, 'dur'] *= 0.1
    df.loc[dos_mask, 'spkts'] *= 5
    
    # Reconnaissance: many connections, low bytes
    recon_mask = df['attack_cat'] == 'Reconnaissance'
    df.loc[recon_mask, 'ct_srv_dst'] *= 5
    df.loc[recon_mask, 'sbytes'] *= 0.2
    
    # Exploits: high payload, specific ports
    exploit_mask = df['attack_cat'] == 'Exploits'
    df.loc[exploit_mask, 'response_body_len'] *= 3
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"unsw_nb15_demo_{n_samples}.csv"
    df.to_csv(sample_path, index=False)
    
    console.print(f"\n[green]✓ Demo dataset created: {sample_path}[/green]")
    console.print(f"  Samples: {len(df):,}")
    console.print(f"  Attack rate: {labels.mean()*100:.1f}%")
    console.print(f"\n  Attack distribution:")
    for cat, pct in attack_types.items():
        count = (attack_cats == cat).sum()
        console.print(f"    {cat}: {count:,} ({100*count/n_samples:.1f}%)")
    
    return sample_path


def main():
    parser = argparse.ArgumentParser(description="Download UNSW-NB15 dataset")
    parser.add_argument("--output", type=Path, default=Path("data/unsw-nb15"))
    parser.add_argument("--sample", type=int, default=50000, help="Sample size")
    parser.add_argument("--kaggle", action="store_true", help="Download from Kaggle")
    args = parser.parse_args()
    
    console.print("\n[bold blue]UNSW-NB15 Dataset Preparation[/bold blue]")
    console.print(f"Output: {args.output}")
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.kaggle:
        if download_from_kaggle(args.output):
            create_sample_from_original(args.output, args.sample)
    else:
        console.print("\n[yellow]Creating demo dataset (UNSW-NB15-like)[/yellow]")
        console.print("For the real dataset, use --kaggle flag")
        create_demo_sample(args.output, args.sample)
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  python scripts/train_all.py --data data/unsw-nb15/unsw_nb15_demo_50000.csv")


if __name__ == "__main__":
    main()
