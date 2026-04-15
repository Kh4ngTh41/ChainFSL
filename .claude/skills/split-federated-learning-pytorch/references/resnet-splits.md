# ResNet-18 Split Points Reference

## Architecture Overview

ResNet-18 structure (torchvision):

```
Input (3, 224, 224)
  ↓
conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
bn1: BatchNorm2d(64)
relu: ReLU
maxpool: MaxPool2d(kernel_size=3, stride=2, padding=1)
  ↓ (64, 56, 56)
layer1: Sequential(BasicBlock × 2)  # 64 channels, no downsampling
  ↓ (64, 56, 56) ← CUT POINT 0
layer2: Sequential(BasicBlock × 2)  # 128 channels, stride=2 in first block
  ↓ (128, 28, 28) ← CUT POINT 1
layer3: Sequential(BasicBlock × 2)  # 256 channels, stride=2 in first block
  ↓ (256, 14, 14) ← CUT POINT 2
layer4: Sequential(BasicBlock × 2)  # 512 channels, stride=2 in first block
  ↓ (512, 7, 7) ← CUT POINT 3
avgpool: AdaptiveAvgPool2d((1, 1))
  ↓ (512, 1, 1)
flatten → (512,)
fc: Linear(512, num_classes)
  ↓ (num_classes,)
Output
```

## Communication Cost Analysis

For batch_size=32, input=(3, 224, 224):

| Cut Point | Activation Shape | Elements | Memory (FP32) |
|-----------|------------------|----------|---------------|
| 0 (layer1) | (32, 64, 56, 56) | 6,422,528 | 24.5 MB |
| 1 (layer2) | (32, 128, 28, 28) | 3,211,264 | 12.3 MB |
| 2 (layer3) | (32, 256, 14, 14) | 1,605,632 | 6.1 MB |
| 3 (layer4) | (32, 512, 7, 7) | 802,816 | 3.1 MB |

**Gradient communication:** Same size as activations (backward pass).

**Total per iteration:** 2× activation size (forward + backward).

## Computational Distribution

Parameter count by section (ResNet-18):

```
conv1 + bn1: 9,472 params
layer1: 147,968 params
layer2: 525,568 params
layer3: 2,099,712 params
layer4: 8,393,728 params
fc: 5,130 params (for 10 classes)

Total: ~11.2M params
```

**Client compute % by cut point:**
- Cut 0: ~1.4% (conv1 + layer1)
- Cut 1: ~6.1% (+ layer2)
- Cut 2: ~25% (+ layer3)
- Cut 3: ~100% (+ layer4, but server only does fc)

## Choosing Cut Points for Different Scenarios

### IoT Edge Devices (Very Low Compute)
**Recommendation:** Cut 0 or Cut 1
- Minimizes client-side FLOPs
- Accepts higher communication cost
- Example: Raspberry Pi, mobile phones on battery

### Privacy-Sensitive Applications
**Recommendation:** Cut 2 or Cut 3
- Deeper layers encode semantic features (harder to reconstruct raw data)
- Lower communication overhead (side benefit)
- Example: Medical imaging, financial data

### Bandwidth-Constrained Networks
**Recommendation:** Cut 2 or Cut 3
- Minimize activation size
- Trade client compute for lower network usage
- Example: Remote sensors, satellite links

### Balanced Collaborative Training
**Recommendation:** Cut 2
- Middle ground for compute, communication, and privacy
- Most common choice in split learning research
- Works well for moderate edge devices (NVIDIA Jetson, etc.)

## Dynamic Cut Layer Selection

For heterogeneous clients:

```python
class AdaptiveSplitResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... initialize all layers ...
    
    def forward_client(self, x, cut_layer):
        """Dynamic cut point based on client capability."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if cut_layer == 0: return x
        
        x = self.layer2(x)
        if cut_layer == 1: return x
        
        x = self.layer3(x)
        if cut_layer == 2: return x
        
        x = self.layer4(x)
        return x
    
    def forward_server(self, x, cut_layer):
        """Resume from dynamic cut point."""
        if cut_layer < 1:
            x = self.layer2(x)
        if cut_layer < 2:
            x = self.layer3(x)
        if cut_layer < 3:
            x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

## ResNet-50 Split Points

For larger models, the pattern is similar but with Bottleneck blocks:

```
layer1: Bottleneck × 3  # 256 channels out
layer2: Bottleneck × 4  # 512 channels out
layer3: Bottleneck × 6  # 1024 channels out
layer4: Bottleneck × 3  # 2048 channels out
```

Activation sizes (batch=32):
- Cut 0: (32, 256, 56, 56) = 25.7 MB
- Cut 1: (32, 512, 28, 28) = 12.9 MB
- Cut 2: (32, 1024, 14, 14) = 6.4 MB
- Cut 3: (32, 2048, 7, 7) = 3.2 MB

Same principles apply, but communication costs are higher due to wider channels.
