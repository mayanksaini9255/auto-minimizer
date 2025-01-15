# Focus-Based Window Minimizer

This Python script automatically minimizes all open windows on your Windows system when it detects that you are not looking at the screen. It leverages a pre-trained machine learning model to identify whether the user is focused on the screen or not.


## Project Overview

The script uses a real-time webcam feed to monitor your gaze. If the model detects that you've looked away from the screen for a sufficient duration, it minimizes all windows to reduce distractions. When gaze returns to the screen, windows are restored.

This project provides a helpful tool for enhancing focus and productivity by minimizing interruptions from open applications.


## Model Details

The model used for gaze detection is a TensorFlow SavedModel (`saved_model.pb`) located in this repository.  The model was trained using Liner, a no code ML model trainer.
The training data consisted of 30-40 images of focused and unfocused images of mine.
The model would be biased as the images as of now are not of any other human, I will train the model again on a wide variety of images later on when I get the time.
Improvement in accuracy is expected with training on a larger, more diverse dataset.

The model classifies each frame into two categories:

* `"focused"`: User is looking at the screen.
* `"unfocused"`: User is looking away from the screen.


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mayanksaini9255/auto-minimizer.git
   ```

2. **Install dependencies:** Navigate to the project directory and install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Pre-trained model:** The pre-trained model (`saved_model.pb`) is included in this repository. Ensure this file is in the correct location (or adjust the path in `focus_minimize.py`).


## Usage

1. **Run the script:** Execute the following command in your terminal:
   ```bash
   python focus_minimize.py
   ```
   The script will start capturing frames from your default webcam.

2. **Gaze Detection:** The script will use the webcam feed to classify your gaze state.

3. **Window Minimization/Restoration:** The script will minimize all windows when it consistently detects "unfocused" gaze and restore them when it detects that you're back to looking at the screen. A time threshold prevents flickering when gaze shifts briefly away from the screen.

4. **Exit:** Press the 'q' key to close the application.


## Configuration (`focus_minimize.py`)

You can adjust the following parameters within the `focus_minimize.py` script to fine-tune the behavior:

* `model_input_width`, `model_input_height`: Adjust if your model requires a different input image size.
* `buffer_size`: The number of recent gaze classifications to consider for smoothing (higher values mean slower responsiveness).
* `unfocused_threshold`: The minimum percentage of "unfocused" classifications needed to trigger window minimization.
* `unfocused_time_threshold`: The minimum number of seconds to remain "unfocused" before window minimization is initiated.


## Dependencies

The project relies on the following Python libraries:

* `opencv-python`
* `pyautogui`
* `tensorflow`
* `numpy`
* `Pillow` (PIL)
* `collections`


## Contributing

Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

