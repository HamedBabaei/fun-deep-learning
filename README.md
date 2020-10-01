# Fun Deep Learning
This is my repository for my works on Deep Learning Specialization course - I am going to make it more fun.

### Courses: 

**Note**: Course label is **`Course1`** for tracking all changes for this week 

* Week 1: **Introduction to deep learning**
* Week 2: **Neural Networks Basics**
     * **Logistic Regression as a Neural Network** and **Python Basics with Numpy**(utils dir). This week label is **`Course1-Week2`** for tracking the changes
     * loss function implementations `L1`, `L2` methods added. (`utils/loss.py`)
     * image utils `image2vector`, `display_image`, `flatten_X`, and `standardize_dataset` methods added. (`utils/image.py`)
     * activation functions `sigmoid` , `softmax`, and `sigmoid_derivative` methods added.(`utils/activation.py`)
     * base utils, `initialize_with_zeros` and `plot_cost_function` added. (`utils/base.py`)
     * dataloader utils for loading different datasets added.(`utils/dataloader.py`)
     * normalization methods `normalize_image` and `normalize_rows` methods added.(`utils/normalization.py`)
     * **LogisticRegression model with Neural Network mindset** a directory with its own files added.
     * Dataset `dataset/catvnotcat` were used for this week added to dataset dir.
* Week 3: **Shallow neural networks**
     * **Planar data classification with one hidden layer** directory
     * `nn.py` consist of nn model for classification of planner data, with an `NN` class
     * `test_nn.py` contains unit tests
     * `util.py` for visualization
     * `nn.ipynb` is a jupyter notebook file for demo of classification
* Week 4: **Deep Neural Networks**
     * **Buidling DNN-Step by Step for CatvsNot Classification** a dictionary with its own files added:
     * `activations.py` contains activations and their backward calculations for DNN
     * `test_dnn.py` contains unit tests
     * `DNN.ipynb` is a jupyter notebook file for demo of dnn classification
     * `dnn.py` is a deep neural network implementations. it contains `DNN` class as a base class and `TwoLayerModel`, `LLayerModel` classes as a child class of `DNN`.
* Week 5: **Practical aspects of Deep Learning**
     * following directories contains `dnn.py`, `activation.py`, utils script, another script for test, and jupyter notebook as a demo
     * `Regularization` directory, L2 regularization and dorpout technique added to `dnn.py`
     * `Initialization` directory, 3 different initialization technique added to `dnn.py`
     * `GradientChecking` directory, added grad `gradcheck_utils.py` for gradient checking technique
* Week 6: **Optimization algorithms**
* Week 7: **Hyperparameter tuning, Batch Normalization and Programming Frameworks**


**Extra and useful links:**
* https://www.cs.ryerson.ca/~aharley/neural-networks/
* https://cs231n.github.io/
* auto-reloading external module: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython





   