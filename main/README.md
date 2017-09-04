# Dilation and Erosion filters in CUDA

Several implementations of the dilation and erosion filters are shown:

* CPU:

* GPU:  


## Performance

I have performed some tests on a Nvidia GTX 1080Ti.

With an image of 320x256 and a radio ranging from 1 to 15:

| Radio / Implementation | Speed-up | CPU | Naïve | Separable | Shared mem. | Radio templatized | Filter op. templatized |
| ---------------------- | -------- | --- | ----- | --------- | ----------- | ----------------- | ---------------------- |
