from ultralytics import YOLO

# Load your custom YOLOv5 model
model = YOLO('best.pt')

# Perform inference on a video
results = model.predict('classico_videoshort.mp4', save=True)

# Print the results
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)