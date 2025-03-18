# Violence Detection Using Deep Learning

## Overview
This project implements a deep learning-based violence detection system using a pre-trained VGG19 model combined with an LSTM network. It processes video frames to detect violent activity and provides real-time alerts when violence is detected.

## Features
- Uses **VGG19** as a feature extractor.
- Employs **LSTM** for sequential data analysis.
- Real-time violence detection on video streams.
- Allows loading pre-trained weights.
- Skips frames to optimize performance.

## Requirements
Ensure you have the following dependencies installed before running the project:

```bash
pip install numpy opencv-python tensorflow scikit-image matplotlib
```

## File Structure
```
│── functions.py  # Contains model definition and helper functions
│── main.py       # Main script to run the violence detection
```

## Usage
To run the program, execute the following command:
```bash
python main.py
```
Ensure that:
- The `myWeights.hdfs` file (pre-trained model weights) is in the correct location.
- The input video file (`input/test1.mp4`) exists.

## Functionality
### `videoFightModel(tf, wight, is_train=False)`
- Loads the VGG19-based model with LSTM for video classification.
- If `is_train=False`, loads pre-trained weights from `wight`.

### `pred_fight(model, video, accuracy=0.9)`
- Predicts whether violence is present in the given video sequence.

### `load_dataset(data_dir)`
- Loads training data from the given dataset directory.

### `test_model(model, X_test, y_test)`
- Evaluates the model performance on test data.

### `stream(filename, model_dir)`
- Streams the video and detects violence in real-time.

## Notes
- The model assumes input videos are preprocessed to 160x160 resolution.
- The model detects violence with a threshold accuracy of **0.967**.
- Press `q` to exit the live video stream.

## License
This project is for educational and research purposes only. Modify and use it at your own discretion.
