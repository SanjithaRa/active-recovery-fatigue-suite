"""
Root wrapper for Standalone Fatigue Inference Pipeline.
Delegates to recovery_backend/test_inference.py.
"""
import os
import sys

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recovery_backend")
sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    from recovery_backend.test_inference import main
    main()
