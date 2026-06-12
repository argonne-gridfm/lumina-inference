"""Example: Load a LUMINA model from Hugging Face and run predictions.

This script demonstrates how to use lumina-inference to:
1. Download model artifacts from Hugging Face via from_pretrained()
2. Load an OPF dataset
3. Run predictions on the dataset

Usage:
    python examples/huggingface_inference.py
"""

import torch

from lumina_inference.modeler import Modeler
from lumina_inference.dataset.opf_dataset import OPFDataset
from lumina_inference.loader.opf_loader import DataLoader

# Initialize device and modeler, then download + load the published model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modeler = Modeler(device)
modeler.from_pretrained("argonne/LUMINA-2M")

# Load dataset and create loader
case_name = modeler.config_data.get("case_name", "pglib_opf_case14_ieee")
dataset = OPFDataset(root="./opf_data", case_name=case_name)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Run predictions (no evaluation)
preds = modeler.run_predictions(loader, max_batches=50)
print(f"Collected predictions for {len(preds)} batches.")

# Inspect first prediction
if preds:
    predictions_cpu, batch_cpu = preds[0]
    print(f"Prediction keys: {list(predictions_cpu.keys())}")
    for key, tensor in predictions_cpu.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
