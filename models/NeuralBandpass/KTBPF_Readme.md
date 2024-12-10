# Keytracking Bandpass Filter Layer

A learnable, adaptive bandpass filter layer for audio deep learning applications. This layer implements a novel approach to frequency filtering that adapts to the input signal characteristics while maintaining trainable parameters for optimization through backpropagation.

## Installation

```bash
pip install torch torchaudio
```

## Quick Start

```python
import torch
from keytracking_filter import KeytrackingBandpassFilter

# Initialize the filter layer
filter_layer = KeytrackingBandpassFilter(
    sample_rate=44100,
    learnable=True
)

# Use in a model
class AudioModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = KeytrackingBandpassFilter()
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3)
        # ... rest of your model
    
    def forward(self, x):
        x = self.filter(x)  # Apply filtering
        return self.conv1(x)  # Continue with normal processing
```

## Parameters and Hyperparameters

### Initialization Parameters

- `sample_rate` (int, default=44100): The sample rate of your audio data
- `min_lowpass_freq` (float, default=0): Minimum frequency for lowpass filter
- `max_lowpass_freq` (float, default=60): Maximum frequency for lowpass filter
- `min_highpass_freq` (float, default=2000): Minimum frequency for highpass filter
- `max_highpass_freq` (float, default=20000): Maximum frequency for highpass filter
- `filter_slope` (float, default=1.0): Initial slope for both filters
- `learnable` (bool, default=True): Whether to enable learnable parameters

### Understanding the Parameters

#### Frequency Bounds
- The lowpass filter operates between `min_lowpass_freq` and `max_lowpass_freq`
- The highpass filter operates between `min_highpass_freq` and `max_highpass_freq`
- Choose these based on your audio domain:
  - For music: Keep `max_lowpass_freq` around 60Hz to filter out sub-bass noise
  - For speech: Can increase `max_lowpass_freq` to ~120Hz
  - Adjust `min_highpass_freq` based on your highest expected fundamental frequency

#### Filter Slope
- Controls how sharp the frequency cutoff is
- Higher values (>1.0) create steeper cutoffs but may introduce ringing
- Lower values (<1.0) create gentler slopes but may let through more noise
- Start with 1.0 and adjust based on your needs

#### Learnable Parameters
When `learnable=True`, the following parameters are optimized during training:
- Lowpass cutoff frequency
- Highpass cutoff frequency
- Filter slope

## Training Guide

### Basic Training Setup

```python
# Initialize model and optimizer
model = AudioModel()
optimizer = torch.optim.Adam([
    {'params': model.filter.parameters(), 'lr': 0.0001},  # Smaller learning rate for filter
    {'params': (p for n, p in model.named_parameters() if 'filter' not in n), 'lr': 0.001}
])

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Training Tips

1. **Learning Rate**: Use a smaller learning rate for filter parameters compared to the rest of the network
   - Filter parameters should change slowly to maintain stability
   - Recommended ratio: 0.1x to 0.01x of main network learning rate

2. **Monitoring Filter Parameters**:
   ```python
   # Print current filter frequencies
   print(f"Lowpass freq: {model.filter.lowpass_freq.item():.2f} Hz")
   print(f"Highpass freq: {model.filter.highpass_freq.item():.2f} Hz")
   ```

3. **Gradient Clipping**: Consider using gradient clipping to prevent large parameter changes
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

## Common Issues and Solutions

1. **Unstable Training**
   - Reduce filter learning rate
   - Enable gradient clipping
   - Check if frequency bounds are appropriate for your data

2. **Over-filtering**
   - Increase filter slope value
   - Widen frequency bounds
   - Reduce learning rate for filter parameters

3. **Under-filtering**
   - Decrease filter slope value
   - Narrow frequency bounds
   - Check if input normalization is appropriate

## Advanced Usage

### Fixed + Learned Parameters
```python
# Initialize with fixed highpass but learnable lowpass
filter_layer = KeytrackingBandpassFilter(
    learnable=True,
    min_highpass_freq=1000,
    max_highpass_freq=1000  # Fixed at 1kHz
)
```

### Custom Initialization
```python
# Initialize with specific starting frequencies
filter_layer = KeytrackingBandpassFilter(
    learnable=True,
    min_lowpass_freq=20,
    max_lowpass_freq=100,
    min_highpass_freq=1500,
    max_highpass_freq=4000
)
```

## Performance Considerations

- The layer adds minimal computational overhead
- Memory usage scales linearly with input size
- Consider using half precision (float16) for large batches
- CPU processing is supported but GPU is recommended

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.