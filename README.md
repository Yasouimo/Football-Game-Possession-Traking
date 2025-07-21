# Football Game Analysis 

## Brief Description
This project is designed to analyze football (soccer) game footage using computer vision and machine learning techniques. It tracks players and the ball, assigns teams to players, and identifies which team has possession of the ball at any given time. The output is an annotated video with visualizations of player movements, team assignments, and ball possession.

---

## Content

### Overview
The project processes video footage of a football game to:
1. **Track Players and the Ball**: Using a pre-trained object detection model, it identifies and tracks players and the ball throughout the video.
2. **Assign Teams**: It assigns a team (e.g., Team A or Team B) to each player based on their jersey color.
3. **Ball Possession Analysis**: It determines which team has possession of the ball at any given time.
4. **Visualization**: The final output is an annotated video with bounding boxes, team colors, and ball possession information.
   ![Image](https://github.com/user-attachments/assets/a73464d7-058c-4dc4-a66c-515a8a240b96)

### Models
1. **Object Detection Model**:
   - A pre-trained YOLO (You Only Look Once) model (`best.pt`) is used to detect and track players and the ball in the video.
   - The model is fine-tuned for football game footage to ensure accurate detection.

2. **Team Assigner**:
   - A custom algorithm assigns teams to players based on their jersey colors.
   - It uses the first frame of the video to determine team colors.

3. **Ball Assigner**:
   - A custom algorithm determines which player is closest to the ball and assigns ball possession to the corresponding team.

### Requirements
To run this project, you need the following Python libraries and tools:

#### Python Libraries
- **OpenCV**: For video processing and visualization.
- **NumPy**: For numerical operations.
- **scikit-learn**: For clustering and distance calculations (if needed).
- **cvzone**: A helper library for OpenCV.
- **supervision**: For annotation and visualization.
- **PyTorch**: For running the YOLO model.

#### Installation
Install the required libraries using `pip`:
```bash
pip install -r requirements.txt
