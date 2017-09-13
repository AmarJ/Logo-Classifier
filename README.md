![Alt text](Kaptur-Logo.png?raw=true "Kaptur_Logo")

Kaptur is a logo recognition tool that can identify 27 different logos present in an image. 

##

Kaptur uses machine learning for the task of detecting logos. When designing my convolutional neural network I use a similar approach to what Matthew Earl uses for his license plate recognition project. His approach can be found [here](https://matthewearl.github.io/2016/05/06/cnn-anpr/). 

I used the [Flickr Logos 27](http://image.ntua.gr/iva/datasets/flickr_logos/) data set to train the CNN on 27 different logos. 

For now the only logos that Kaptur can identify are the following: Adidas, Apple, BMW, Citroen, Coca Cola, DHL, Fedex, Ferrari, Ford, Google, Heineken, HP, McDonalds, Mini, Nbc, Nike, Pepsi, Porsche, Puma, Red Bull, Sprite, Starbucks, Intel, Texaco, Unisef, Vodafone and Yahoo.

*Update: For logo recognition I have now switched to the YOLO (You only look once) approach using the [Darknet framework](https://github.com/pjreddie/darknet). [YOLO](https://arxiv.org/pdf/1506.02640.pdf) is a phenomenal real-time object detection system created by [Joseph Redmon](https://arxiv.org/find/cs/1/au:+Redmon_J/0/1/0/all/0/1). The problem with my logo recognition tool is that it's just a classifier where it only recognizes a logo if the logo is the only thing in the image. I added a bunch stuff around it like Kaptur GraphCut to repurpose my classifier into a detection system. YOLO's approach however, uses a single neural network to predict bounding boxes and class probabilities directly from full images in one evaluation. 

You can see the results of this project [here](https://github.com/AmarJ/darknet-NN-framework).
