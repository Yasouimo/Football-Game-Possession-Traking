# import sys
# sys.path.append('../')
# from utils import get_center_of_bbox, measure_distance
#
#
# class Player_ball_assigner():
#     def __init__(self):
#         self.distance_threshold=70
#
#         #we gonna create a function for one frame and in the main we gonna loop
#     def ball_assigner(self, ball_box,players):
#
#         centerBall=get_center_of_bbox(ball_box)
#         minimum_distance=90000
#         for id, player in players.items():
#             playerbb=player['bbox']
#             assignedplayer = -1
#             distance_left = measure_distance((playerbb[0], playerbb[-1]), centerBall)
#             distance_right = measure_distance((playerbb[2], playerbb[-1]), centerBall)
#             distance = min(distance_left, distance_right)
#             if distance<self.distance_threshold:
#                 if distance<minimum_distance:
#                     assignedplayer=id
#                     minimum_distance=distance
#             return assignedplayer
#

import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class Player_ball_assigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def ball_assigner(self,ball_bbox , players ):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player



