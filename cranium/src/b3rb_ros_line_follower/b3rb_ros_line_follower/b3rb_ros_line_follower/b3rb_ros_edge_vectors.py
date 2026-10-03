#!/usr/bin/env python3
# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10

# BGR Color Constants
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# ROI Configuration
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.50
VECTOR_MAGNITUDE_MINIMUM = 2.25
MIN_CONTOUR_AREA = 40.0
BLACK_THRESHOLD_VAL = 45


class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node for the NXP CUP Autonomous Vehicle Challenge.
    Processes camera images to extract track boundary vectors (left/right lines),
    computes the center midpoint trajectory vector, and publishes EdgeVectors for steering control.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscriber: Compressed camera stream
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT
        )

        # Publisher: Processed edge vectors for line follower node
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT
        )

        # Publisher: Debug binary threshold mask
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            '/debug_images/thresh_image',
            QOS_PROFILE_DEFAULT
        )

        # Publisher: Debug BGR image with drawn vectors
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            '/debug_images/vector_image',
            QOS_PROFILE_DEFAULT
        )

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

    def publish_debug_image(self, publisher, image):
        """Helper to compress and publish debug OpenCV images."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def compute_vectors_from_image(self, thresh):
        """
        Extracts valid boundary line vectors from the binary thresholded image mask.
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        vectors = []

        for contour in contours:
            # Filter noise by area
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            coordinates = contour[:, 0, :]

            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = coordinates[coordinates[:, 1] == min_y_value]
            max_y_coords = coordinates[coordinates[:, 1] == max_y_value]

            min_y_coord = np.array(min_y_coords[0], dtype=np.float64)
            max_y_coord = np.array(max_y_coords[0], dtype=np.float64)

            # Vector length validation
            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if magnitude > VECTOR_MAGNITUDE_MINIMUM:
                rover_point = np.array([self.image_width / 2.0, self.lower_image_height], dtype=np.float64)
                middle_point = (min_y_coord + max_y_coord) / 2.0
                distance = np.linalg.norm(middle_point - rover_point)

                vectors.append([min_y_coord, max_y_coord, distance])

        return vectors

    def process_image_for_edge_vectors(self, image):
        """
        Processes frame ROI, extracts left/right vectors, calculates track midpoint,
        and draws visual debug indicators on the output image.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = self.image_height - self.lower_image_height

        # 1. Crop ROI (lower portion of the image)
        image_cropped = image[self.upper_image_height:].copy()

        # 2. Preprocessing & Shadow Rejection
        gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh_cropped = cv2.threshold(blurred, BLACK_THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)

        # 3. Vector Extraction & Distance Sorting
        raw_vectors = self.compute_vectors_from_image(thresh_cropped)
        raw_vectors.sort(key=lambda x: x[2])

        # 4. Classify vectors into Left and Right pools based on image center
        half_width = self.image_width / 2.0
        vectors_left = [v for v in raw_vectors if ((v[0][0] + v[1][0]) / 2.0) < half_width]
        vectors_right = [v for v in raw_vectors if ((v[0][0] + v[1][0]) / 2.0) >= half_width]

        selected_vectors = []
        left_vector = None
        right_vector = None

        if len(vectors_left) > 0:
            left_vector = vectors_left[0]
            selected_vectors.append(left_vector)

        if len(vectors_right) > 0:
            right_vector = vectors_right[0]
            selected_vectors.append(right_vector)

        # 5. Visualization on Debug Image (Green Boundary Lines)
        debug_img = image_cropped.copy()
        for vec in selected_vectors:
            pt1 = (int(vec[0][0]), int(vec[0][1]))
            pt2 = (int(vec[1][0]), int(vec[1][1]))
            cv2.line(debug_img, pt1, pt2, GREEN_COLOR, 3)

        # 6. Calculate & Draw Midpoint Trajectory Vector (Blue Center Line)
        if left_vector is not None and right_vector is not None:
            mid_min_x = (left_vector[0][0] + right_vector[0][0]) / 2.0
            mid_min_y = (left_vector[0][1] + right_vector[0][1]) / 2.0
            mid_max_x = (left_vector[1][0] + right_vector[1][0]) / 2.0
            mid_max_y = (left_vector[1][1] + right_vector[1][1]) / 2.0

            mid_pt1 = (int(mid_min_x), int(mid_min_y))
            mid_pt2 = (int(mid_max_x), int(mid_max_y))
            cv2.line(debug_img, mid_pt1, mid_pt2, BLUE_COLOR, 2)

        # 7. Map ROI Y-coordinates back to global camera frame coordinates
        final_vectors = []
        for vec in selected_vectors:
            p1 = [vec[0][0], vec[0][1] + self.upper_image_height]
            p2 = [vec[1][0], vec[1][1] + self.upper_image_height]
            final_vectors.append([p1, p2])

        # Publish Debug Images
        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        return final_vectors

    def camera_image_callback(self, message):
        """Callback to process incoming ROS compressed images and publish EdgeVectors."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        vectors = self.process_image_for_edge_vectors(image)

        # Construct ROS 2 payload
        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Assign Vector 1 (Left Boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Assign Vector 2 (Right Boundary)
        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)


def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()