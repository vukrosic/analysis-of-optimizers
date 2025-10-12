# I tried mixing DeltaNet and Attention and found something surprising

I trained 13 language models from 0% to 100% attention (rest being DeltaNet linear attention). Pure Transformer with 100% attention was the WORST performer - ranked dead last. The winner? Just 17% attention (2 layers out of 12), which beat pure Transformer by 27% and pure DeltaNet by 8%.

Turns out the optimal balance isn't 50/50 or 100% - it's heavily skewed toward DeltaNet with strategic attention layers. Less really is more.

