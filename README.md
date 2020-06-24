# DAB: Differentiable Approximation Bridges

A simplified example demonstrating a DAB network presented in [Improving Discrete Latent Representations With Differentiable Approximation Bridges](https://arxiv.org/abs/1905.03658).

### Usage

The only dependency for this demo is [pytorch](https://pytorch.org/get-started/locally/).  
To run the 10-sort signum-dense problem described in section 4.1 of the [paper](https://arxiv.org/abs/1905.03658) simply run:

```python
python main.py
```

This should result in the following which corroborates the paper’s result of 94.2% :

```bash
train[Epoch 2168][1999872.0 samples][7.79 sec]: Loss: 79.2356   DABLoss: 7.9058 Accuracy: 95.5683
…
test[Epoch 2168][399360.0 samples][0.91 sec]: Loss: 79.2329     DABLoss: 7.9012 Accuracy: 94.6424
```

### Create a DAB for a custom non-differentiable function

  1. Create a suitable approximation neural network.
  2. Implement custom hard function similar to SignumWithMargin in models/dab.py .
  3. Stack a DAB module in your neural network pipeline.
  4. Add DAB loss to normal loss.


### Cite

```
@article{
  dabimprovingdiscreterepr2020,
  title={Improving Discrete Latent Representations With Differentiable Approximation Bridges},
  author={Ramapuram, Jason and Webb, Russ},
  journal={IEEE WCCI},
  year={2020}
}
