In demosaic.py I have implemented all features required by the assignment description.

## How to run:

```
pip3 install opencv-python
python3 demosaic.py
```

## Part 1

I have zoomed in into the head of one of the pencils. It seems that the artifacts occure the most
in areas of high contrast / detail. The effect also seems to be amplified when the area of high
details spans an extended distance / area. I believe the main cause of this effect is that the 
details areas have a big change in color over a small portion of the image. Since two out of
the three color channels are interpolated from the neighboring pixels, one would expect those
interpolated values to be less accurate since the inherent innacuracy is amplified by the contrast.

## Part 2

When using Freeman's technique, there is a definite improvement in the color representation. Some
high-contrast areas become discolored using basic interpolation, but with Freeman's technique the
overall color seems to be visually more accurate. That said, when looking at the root of the squared
differences merged into a single grayscale channel, the improvement is much less apparent. 