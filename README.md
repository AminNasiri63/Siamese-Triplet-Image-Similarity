# 🧠 Image Similarity Estimation using Siamese Network with Triplet Loss

## Overview

This project implements a Siamese neural network trained using **triplet loss** for **image similarity estimation**. The model learns to embed images into a feature space where similar images are close together and dissimilar images are far apart. The model is built in TensorFlow/Keras and leverages distributed training across GPUs for faster convergence.  
The project includes data handling, model creation, training with freezing and fine-tuning, and detailed evaluation metrics.

## Key Features
 ✅ TensorFlow `tf.data` pipeline for efficient data loading.  
 ✅ Distributed training using `tf.distribute.Strategy.  
 ✅ Modular and scalable training script.  
 ✅ Data pipeline using triplet generators.  
 ✅ Triplet loss function (anchor, positive, negative).  
 ✅ Progressive training (frozen backbone → fine-tuning).  
 ✅ Using transfer learning with ResNet50 for embeddings.  
 ✅ Comprehensive evaluation with confusion matrices and per-class metrics.  
 ✅ Visualization of training loss and evaluation results.  
 ✅ Easily adaptable for your own image datasets.  

## Data Structure

Your dataset directory should be organized as follows:

```
root_dataset_dir/
│
├── ID_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── ID_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── ID_N/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

- Each subfolder corresponds to one unique identity/class (`ID_1`, `ID_2`, …).
- Each subfolder contains multiple images of that identity.
- The code selects triplets (anchor, positive, negative) based on these folders.

## Installation
Make sure you have Python 3.7+ and the following packages installed:

```
pip install tensorflow seaborn matplotlib numpy pandas
```

## Usage
* **Prepare your dataset:**
Organize your image dataset as described above.

* **Configure paths and parameters:**
Edit the pathData variable and other parameters inside the notebook as needed.

* **Run training:**
Execute the training notebook to train the Siamese network.

* **Evaluate the model:**
Use the provided evaluation functions and cells to compute similarity metrics and visualize results.

## Project Structure
* **SiameseNetwork.ipynb:** Jupyter notebook containing the full pipeline (data loading, training, evaluation)

## References

- [Keras Example: Siamese Network](https://keras.io/examples/vision/siamese_network/)
- [TensorFlow Guide: Distributed Training with Keras](https://www.tensorflow.org/tutorials/distribute/keras)
- [TensorFlow Guide: Distributed Training Overview](https://www.tensorflow.org/guide/distributed_training)
- [Keras Guide: Transfer Learning & Fine-tuning](https://keras.io/guides/transfer_learning/)

## License
MIT License

