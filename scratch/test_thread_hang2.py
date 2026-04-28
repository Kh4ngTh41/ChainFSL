import sys
import os
import torch
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chainfsl.src.sfl.models import SplittableResNet18
from chainfsl.src.sfl.trainer import SFLTrainer

LOG_FILE = '/mnt/f/ChainFSL/scratch/hang_log2.txt'

def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def test_single_thread(model, node_id):
    log(f"[{node_id}] Creating trainer...")
    trainer = SFLTrainer(node_id=node_id, model=model, cut_layer=2, device=torch.device('cpu'))
    
    log(f"[{node_id}] Creating dummy data...")
    inputs = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 10, (8,))
    
    log(f"[{node_id}] Doing local step...")
    
    log(f"[{node_id}] Client forward...")
    smash_data = trainer.client.forward(inputs)
    
    log(f"[{node_id}] Server forward_backward...")
    loss, grad = trainer.server.forward_backward(smash_data, labels)
    
    log(f"[{node_id}] Client backward...")
    trainer.client.backward(grad)
    
    log(f"[{node_id}] Done local step. Loss={loss:.4f}")
    return True

if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    # log("Setting num threads to 1")
    # torch.set_num_threads(1)
    
    log("Creating global model...")
    model = SplittableResNet18(n_classes=10, cut_layer=2).cpu()
    
    workers = 4
    log(f"Starting ThreadPoolExecutor with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(workers):
            futures.append(executor.submit(test_single_thread, model, i))
            
        for f in as_completed(futures):
            log(f"Future completed: {f.result()}")
    log("All done!")
