Uses a custom version of the SIFT algorithm to match keypoints between images, then uses those keypoints to compute a matrix for projecting the images into eachothers coordinate frame and then projects the images into a single large image, statching them together to produce a panorama. below is an example output from three photos I took in the dolomites in Italy:

## Example

### Inputs

![image](https://github.com/Davidster/MiniSIFT/assets/2389735/fe212b00-cf25-4a67-a740-a4dbb6e0bd4a)

![image](https://github.com/Davidster/MiniSIFT/assets/2389735/7dce596b-edbd-46c8-a5fb-fc5e751272c4)

![image](https://github.com/Davidster/MiniSIFT/assets/2389735/102dd861-beb7-4bce-9c68-6458baea7478)

### Output

![image](https://github.com/Davidster/MiniSIFT/blob/master/rust/images/dolomites_panorama.jpg)
