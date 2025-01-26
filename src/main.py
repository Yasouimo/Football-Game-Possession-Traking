from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import Player_ball_assigner

def main():
    # Read Video and store all the frames in a list
    video_frames = read_video('input videos/goodview.mp4')
    print(f"Total frames in video: {len(video_frames)}")  # Debug: Check video frames

    tracker = Tracker(model_path='models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='tracksS/track_stubs.pkl')
    print(f"Total frames in tracks['players']: {len(tracks['players'])}")  # Debug: Check tracks frames

    tracks["ball"] = tracker.ball_interpolation(tracks["ball"])

    # Assign a color to each team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        print(f"Processing frame {frame_num}")  # Debug: Print current frame number
        if frame_num >= len(video_frames):
            break  # Exit if frame_num exceeds the number of video frames
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            if team is not None:  # Ensure team is assigned
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            else:
                print(f"Warning: Player {player_id} in frame {frame_num} could not be assigned a team")

    # Debug: Check if all players have teams
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            if 'team' not in track:
                print(f"Warning: Player {player_id} in frame {frame_num} has no team assigned")

    ball_player_assigned = []
    ballplayerOb = Player_ball_assigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ballbb = tracks["ball"][frame_num][1]['bbox']
        assigned_player = ballplayerOb.ball_assigner(ballbb, player_track)
        print(f"Assigned player: {assigned_player}")  # Debug: Check assigned player

        if assigned_player != -1:
            if assigned_player in tracks['players'][frame_num]:
                if 'team' in tracks['players'][frame_num][assigned_player]:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    ball_player_assigned.append(tracks['players'][frame_num][assigned_player]["team"])
                else:
                    print(f"Warning: 'team' key not found for player {assigned_player} in frame {frame_num}")
            else:
                print(f"Warning: Player {assigned_player} not found in frame {frame_num}")

    team_ball_control = np.array(ball_player_assigned)

    output_videos_labeled = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save the output video
    output_path = "output vidos/output-video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for .avi files
    fps = 30  # Adjust FPS as needed
    frame_size = (output_videos_labeled[0].shape[1], output_videos_labeled[0].shape[0])  # Get frame size
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in output_videos_labeled:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    main()