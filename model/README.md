The trained model can be found [here](https://huggingface.co/nikhil-405/Gray2Color).

The model has a total of 13.4 million parameters and is based on a U-Net architecture with the following hyperparameters:
- **Input Image Size**: 256x256 pixels
- **Batch Size**: 16
- **Learning Rate**: 0.0002
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Number of Epochs**: 50
- **Data Augmentation**: Random horizontal flips, random rotations, and random brightness adjustments
- **Normalization**: Images are normalized to the range [0, 1] before training
- **Activation Function**: ReLU for hidden layers, Sigmoid for output layer
- **Output Channels**: 2 (for the a* and b* channels in the Lab color space)
- **Input Channels**: 1 (for the grayscale channel)
- **Model Type**: U-Net with skip connections
- **Pre-trained Weights**: Available for download from the Hugging Face repository
- **Training Dataset**: The model is trained on the COCO dataset, which contains a wide variety of images for robust colorization performance.