import subprocess

def test_run_experiment_accepts_baseline_flag():
    """run_experiment.py should accept --baseline fedavg."""
    result = subprocess.run(
        ["python3", "experiments/run_experiment.py", "--help"],
        capture_output=True, text=True, cwd="/mnt/f/ChainFSL/chainfsl"
    )
    assert "--baseline" in result.stdout, f"--baseline flag not found in help. stdout: {result.stdout[:500]}"
    assert "--cluster_ratio" in result.stdout, f"--cluster_ratio flag not found"
    print("PASS: --baseline and --cluster_ratio flags available")