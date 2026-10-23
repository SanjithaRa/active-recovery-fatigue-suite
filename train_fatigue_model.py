"""
Root wrapper for WESAD Fatigue Model Training Pipeline.
Delegates to recovery_backend/train_fatigue_model.py.
"""
import os
import sys

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recovery_backend")
sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    from recovery_backend.train_fatigue_model import main
    main()
