# Watch a ConvNet Learn to Distinguish Circles from Squares

![Learning visualization](images/convnet_learning.gif?raw=true)

I implemented a single-layer, single-filter [ConvNet](https://en.wikipedia.org/wiki/Convolutional_neural_network) to classify hand-drawn circles and squares from the [*Quick, Draw!* Dataset](https://github.com/googlecreativelab/quickdraw-dataset).
I then looked at how the filter changed throughout the learning process, and how learning was affected by the learning rate and momentum parameter.

## Code instructions

Want to make your own visualizations?

0. Clone this repo.
1. Install the SciPy stack (Matplotlib, NumPy, etc.), scikit-learn, and TensorFlow.
2. Download `full-numpy_bitmap-circle.npy` and `full-numpy_bitmap-square.npy` from [Google Cloud Storage](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap).
3. Run `main.py` to generate GIF frames.
4. Convert the generated frames into a GIF, e.g., using ImageMagick: `convert -gravity east -extent 800x640 -delay 12 -loop 0 *.png convnet_learning.gif`.

## Credits and license

The training data is made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.

The code in this repository is released under the GNU GPLv3.
