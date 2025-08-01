import torch
import numpy as np
from REIN_ViT import REIN_ViT

def test_model():
    print("Testing REIN_ViT model...")
    
    # Create model
    model = REIN_ViT()
    print(f"✓ Model created successfully")
    print(f"✓ Local feat dim: {model.local_feat_dim}")
    print(f"✓ Global feat dim: {model.global_feat_dim}")
    
    # Test with dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 201, 201)
    print(f"✓ Input shape: {x.shape}")
    
    # Forward pass
    try:
        local_feats, _, global_desc = model(x)
        print(f"✓ Forward pass successful")
        print(f"✓ Local features shape: {local_feats.shape}")
        print(f"✓ Global descriptor shape: {global_desc.shape}")
        
        # Check dimensions
        assert local_feats.shape == (batch_size, 128, 224, 224), f"Local features shape mismatch: {local_feats.shape}"
        assert global_desc.shape == (batch_size, 8192), f"Global descriptor shape mismatch: {global_desc.shape}"
        print("✓ All dimensions correct!")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 All tests passed! Model is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the model.") 