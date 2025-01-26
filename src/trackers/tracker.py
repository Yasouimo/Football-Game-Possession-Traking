import time
import supervision as sv
from ultralytics import YOLO
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import cv2
import sys
import cvzone

# we gonna go out the folder so that python find utils package
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position,get_bbox_height


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def ball_interpolation(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        """"
        x.get(1, {}): This tries to get the value associated with the key 1 from each detection dictionary x. If 1 does not exist, it returns an empty dictionary.
        .get('bbox', []): This retrieves the 'bbox' key from the resulting dictionary. If 'bbox' doesn't exist, it returns an empty list.
        """

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        """""
        r
Copier le code
   x1     y1     x2     y2
0  100.0  150.0  50.0  50.0
1  200.0  250.0  60.0  60.0
2    NaN    NaN    NaN    NaN  # Row for empty list
3  300.0  350.0  70.0  70.0
"""
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        """"
        
        |   A   |   B   |
        |-------|-------|
        |   1   |   4   |
        |  NaN  |   5   |
        |   3   |  NaN  |
        |  NaN  |   7   |
        after:
        |   A   |   B   |
        |-------|-------|
        |   1   |   4   |
        |   3   |   5   |
        |   3   |   7   |
        |  NaN  |   7   |


        """""
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        """"""""""
        args : 
         frames: list containing all the frames of the video

         return: 
         list of all the detections for all the frames in the video 
        """""""""
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            # concatenation of all the predictions in one list
            detections += detection_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # def get_tracks(self, frames, read_from_stub=False, stub_path=None):
    #     """this will be run ssi we don't have already run it and we dnt have the tracks saved"""
    #     if read_from_stub and stub_path is not None and os.path.exists(stub_path):
    #         with open(stub_path, 'rb') as f:
    #             tracks = pickle.load(f)
    #         return tracks
    #
    #     detections = self.detect_frames(frames)
    #     tracks = {
    #         "players": [],
    #         "referees": [],
    #         "ball": [],
    #     }
    #
    #     for frame_nbr, detections in enumerate(detections):
    #
    #         classe_names = detections.names
    #         """"""""""
    #         classe names est de la forme dictionnaire nbr 0: ball 7na bghina ball : 0
    #         """""""""
    #         inv_classe_names = {v: s for s, v in classe_names.items()}
    #         #print(f'normal{classe_names} inverted {inv_classe_names}')
    #
    #         """""""""""
    #         convert the ultralytics formats into the sv format
    #         """""""""""
    #         detections_supervision = sv.Detections.from_ultralytics(detections)
    #         for object_ind, class_id in enumerate(detections_supervision.class_id):
    #             if classe_names[class_id] == "goalkeeper":
    #                 detections_supervision.class_id[object_ind] = inv_classe_names["player"]
    #         """""""Tracking code starts from here"""""""
    #         detectionsWithTracking = self.tracker.update_with_detections(detections_supervision)
    #         # ghi kayzidlo track id ldetections dyal ultralytics
    #         # print(detectionsWithTracking)
    #         # remember this is for each frame
    #         tracks["players"].append({})  # appending the list of the dictionnary with a dictionnary
    #         tracks["referees"].append({})
    #         tracks["ball"].append({})
    #         for frame_detection in detectionsWithTracking:
    #             bbox = frame_detection[0].tolist()
    #             cls_id = frame_detection[3]
    #             track_id = frame_detection[4]
    #             # print(frame_detection) #go through all the detecitons in one frame is an array contaning every thing we need
    #             # print(" new element in detectionsWithTracking/n")
    #             if cls_id == inv_classe_names['player']:
    #                 tracks["players"][frame_nbr][track_id] = {"bbox": bbox}
    #
    #                 """"""""""
    #                 assignment of a new index lihowa track id  and filling the dictionary
    #
    #                 "referees": [
    #     {
    #         1: {"bbox": [x1, y1, x2, y2]},  # Bounding box for referee with track_id=1 in frame 0
    #         2: {"bbox": [x1, y1, x2, y2]},  # Bounding box for referee with track_id=2 in frame 0
    #     },
    #                 """""""""
    #
    #             if cls_id == inv_classe_names['referee']:
    #                 tracks["referees"][frame_nbr][track_id] = {"bbox": bbox}
    #
    #         for frame_detection in detectionsWithTracking:
    #             bbox = frame_detection[0].tolist()
    #             cls_id = frame_detection[3]
    #
    #             if cls_id == inv_classe_names['ball']:
    #                 tracks["ball"][frame_nbr][1] = {"bbox": bbox}
    #         """""just to save the tracks so that we don't need to wait for all this to be done each time"""
    #         if stub_path is not None:
    #             with open(stub_path, 'wb') as f:
    #                 pickle.dump(tracks, f)
    #
    #     return tracks
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cvzone.putTextRect(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1380, 900),  2,
                    3, (255, 255, 255), cv2.FONT_HERSHEY_SIMPLEX)
        cvzone.putTextRect(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1380, 950), 2,
                    3, (255, 255, 255), cv2.FONT_HERSHEY_SIMPLEX)

        return frame
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        height=get_bbox_height(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        cvzone.cornerRect(
            frame,  # The image to draw on
            (int(bbox[0]), int(bbox[1]), int(width), int(height)),  # The position and dimensions of the rectangle (x, y, width, height)
            l=9,  # Length of the corner edges
            t=3,  # Thickness of the corner edges
            rt=1,  # Thickness of the rectangle
            colorR=color,  # Color of the rectangle
            colorC=color# Color of the corner edges
        )

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks,ball_player_assigned):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            # mazal khasni ndrawi dlgardient

            # Draw Players
            for track_id, player in player_dict.items():
                #it will see in the dictionnary if there is a id called team color if not it will give it  RED
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, ball_player_assigned)

            output_video_frames.append(frame)

        return output_video_frames




