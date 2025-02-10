# Eye Tracking Program

This is a simple eye tracking program that detects and monitors eye movement using OpenCV and dlib. It calculates the Eye Aspect Ratio (EAR) to identify suspicious activity based on eye behavior. This project is a starting point, and I plan to develop it further with additional features.

## Features
- Detects facial landmarks using dlib's pre-trained model.
- Tracks eye movement and calculates EAR.
- Identifies potential suspicious behavior based on eye closure duration.
- Provides real-time visualization of detected eye landmarks.

## Requirements
Before running the program, install the necessary dependencies:

```bash
pip install opencv-python numpy dlib
```

Make sure to download the required facial landmark model:

- shape_predictor_68_face_landmarks.dat

Extract the file and update the path in the script accordingly.

## How to Run :

1. Clone this repository:
   
   ```bash
   git clone https://github.com/yourusername/eye-tracking.git
   cd eye-tracking
   ```
   
2. Ensure your webcam is connected.

3. Run the script :

   ```bash
   python eye_tracking.py
   ```

4. Press Esc to exit the program.

## Future Improvements

- Improve accuracy by fine-tuning EAR thresholds.
- Implement head pose estimation for better tracking.
- Add logging to store suspicious activity data.
- Explore deep learning-based eye tracking for better performance.

## Dataset Information

This project utilizes the shape_predictor_68_face_landmarks.dat model from dlib, which was trained on the iBUG 300-W dataset. This dataset contains annotated facial landmarks for face detection and recognition tasks.

For more details on the dataset:

iBUG 300-W Dataset
Dlib Landmark Model

## Contributing

Feel free to contribute by opening issues or submitting pull requests. Any suggestions or improvements are welcome!
