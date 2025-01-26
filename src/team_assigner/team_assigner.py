""""""""""
                            +-------------------+
                            |   TeamAssigner    |
                            +-------------------+
                                     |
                                     |
                    +----------------+----------------+
                    |                                 |
         +---------------------+            +-----------------------+
         |   __init__ method   |            |  get_clustering_model |
         +---------------------+            +-----------------------+
         | Initializes team    |            | - Reshapes image to   |
         | colors and player   |            |   a 2D array         |
         | team dictionary     |            | - Applies KMeans     |
         +---------------------+            |   clustering (2 clusters) |
                    |                      +-----------------------+
                    |                                 |
          +---------+                                |
          |                                          |
    +------------+                                  |
    | assign_team_color                             |
    +------------+                                  |
    | - Extracts each player's color                |
    |   using get_player_color                      |
    | - Creates KMeans clusters for all             |
    |   player colors                               |
    | - Assigns and stores the cluster centers      |
    |   as team colors in team_colors dictionary    |
    +------------+                                  |
          |                                         |
+----------+                               +-------------------------+
|          |                               | get_player_color method |
| get_player_team                          +-------------------------+
| method                                   | - Crops player from     |
+--------------------------+               |   frame                 |
| Checks if player is in   |               | - Gets top half of      |
| player_team_dict         |               |   image                 |
| - If yes, returns        |               | - Clusters top half     |
|   team ID                |               | - Identifies player's   |
| - If no:                 |               |   cluster and color     |
|   1. Calls               |               +-------------------------+
|      get_player_color    |                       |
|   2. Predicts team       |                       |
|      using self.kmeans   |                       |
|   3. Stores player team  |                       |
|      ID in player_team_dict                       |
+--------------------------+

"""""""""


from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        # Initialize team colors and player-team dictionary
        self.team_colors = {}  # Stores the average color for each team
        self.player_team_dict = {}  # Maps player IDs to their assigned team
        self.kmeans = None  # KMeans model for clustering player colors

    def get_clustering_model(self, image):
        """
        Initialize the clustering model for the given image.
        Args:
            image: The image to cluster.
        Returns:
            kmeans: A trained KMeans model with 2 clusters.
        """
        # Reshape the image to a 2D array (pixels x RGB)
        image_2d = image.reshape(-1, 3)

        # Perform KMeans clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant color of a player from their bounding box.
        Args:
            frame: The frame containing the player.
            bbox: The bounding box of the player (x1, y1, x2, y2).
        Returns:
            player_color: The dominant color of the player (RGB).
        """
        # Crop the player from the frame using the bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Take only the top half of the image to avoid noise from shorts/shoes
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get the clustering model for the top half image
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Determine the player's cluster by checking the corners
        corner_clusters = [
            clustered_image[0, 0],  # Top-left corner
            clustered_image[0, -1],  # Top-right corner
            clustered_image[-1, 0],  # Bottom-left corner
            clustered_image[-1, -1]  # Bottom-right corner
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Get the player's dominant color from the cluster center
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assign team colors based on player colors in the first frame.
        Args:
            frame: The first frame of the video.
            player_detections: A dictionary of player detections in the frame.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Cluster player colors into 2 teams using KMeans
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Store the trained KMeans model and team colors
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Get the team ID for a player based on their color.
        Args:
            frame: The frame containing the player.
            player_bbox: The bounding box of the player.
            player_id: The ID of the player.
        Returns:
            team_id: The team ID (1 or 2) for the player.
        """
        # If the player is already assigned to a team, return their team ID
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player's dominant color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team ID using the trained KMeans model
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Convert to team 1 or 2 (instead of 0 or 1)

        # Handle edge cases (e.g., player_id == 91)
        if player_id == 91:
            team_id = 1

        # Store the player's team ID in the dictionary
        self.player_team_dict[player_id] = team_id

        return team_id