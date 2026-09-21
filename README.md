# Cat vs. Dog Image Classifier

A convolutional neural network that classifies images as either a cat or a dog, built with TensorFlow/Keras as my first real computer vision project.

## What it does

Given an image, the model predicts whether it shows a cat or a dog, along with a confidence score. It's trained on the classic ["Dogs vs. Cats"](https://www.kaggle.com/datasets/pybear/cats-vs-dogs) dataset (~25,000 images) using a CNN built from scratch — no pretrained models, no transfer learning, just Conv2D and MaxPooling layers stacked up and trained from zero.

## Result

**88% validation accuracy**, after fixing an overfitting problem in the first version.

## The process (and what went wrong along the way)

This project turned out to be less about the model and more about everything around it:

- **The dataset had corrupt files.** A handful of images crashed training partway through with decode errors. I wrote a cleaning script using Pillow to catch and remove broken files before training, and added `tf.data.experimental.ignore_errors()` as a second safety net for corruption that slips past a basic validity check.
- **The first model overfit badly.** Training accuracy hit 99% while validation accuracy plateaued around 83% — the model had memorized the training images instead of learning general patterns.
- **Fixed it with three techniques together:** data augmentation (random flips, rotation, zoom) so the model never sees the exact same image twice, dropout (50%) to stop it from over-relying on specific features, and early stopping to halt training once validation accuracy stopped improving. That combination closed the gap between training and validation accuracy and pushed validation accuracy up to 88%.

## Tech stack

- Python
- TensorFlow / Keras
- Pillow (dataset cleaning)
- Matplotlib (training visualization)

## Files

| File | What it does |
|---|---|
| `clean_dataset.py` | Scans the dataset and removes corrupted image files before training |
| `cat_dog_classifier.py` | Loads the data, builds the CNN, trains it, and saves the model |
| `test_ai.py` | Loads the saved model and predicts on a new, unseen image |

## What I'd try next

- Transfer learning with a pretrained model (e.g. MobileNet) to compare against training from scratch
- A simple web interface to upload and classify a photo without touching the code
