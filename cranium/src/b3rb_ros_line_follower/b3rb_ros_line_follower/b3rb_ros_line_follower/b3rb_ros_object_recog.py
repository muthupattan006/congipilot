# Copyright 2024-2026 NXP
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
#
# =====================================================================
# REFACTOR NOTES (Pure PyTorch & Torchvision Version)
# =====================================================================
# Removed Ultralytics dependency to strictly comply with NXP Cup India 2026
# pre-installed package list (torch==2.3.0, torchvision==0.18.0).
#
# Inference engine uses torch / torchvision directly:
#   - Images preprocessed using torchvision transforms (Resize, ToTensor)
#   - Inference executed via PyTorch / TorchScript JIT engine
#   - Bounding boxes filtered using torchvision.ops.nms
# =====================================================================

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import torch
import torchvision
import torchvision.transforms as T

QOS_PROFILE_DEFAULT = 10

# Exact path required by the deployment environment.
MODEL_PATH = (
    '/home/bates/cognipilot/cranium/src/b3rb_ros_line_follower/'
    'b3rb_ros_line_follower/b3rb_ros_line_follower/best.pt'
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45  # Non-Maximum Suppression (NMS) threshold

# Standard class mapping for NXP sign boards
# Adjust this dictionary if your model's class indices differ.
CLASS_NAMES: Dict[int, str] = {
    0: 'A_LEFT',
    1: 'A_RIGHT',
    2: 'B_LEFT',
    3: 'B_RIGHT',
    4: 'C_LEFT',
    5: 'C_RIGHT',
    6: 'X_LEFT',
    7: 'X_RIGHT',
    8: 'Y_LEFT',
    9: 'Y_RIGHT',
    10: 'Z_LEFT',
    11: 'Z_RIGHT'
}

# A Detection is (bbox_center_x, class_id, confidence, [x1, y1, x2, y2]).
Detection = Tuple[float, int, float, List[float]]


class ObjectRecognizer(Node):
    """
    Detects navigation sign boards using a PyTorch model and
    publishes a compact "A:LEFT;B:RIGHT" style string describing every
    label visible on the board, sorted left-to-right.
    """

    def __init__(self) -> None:
        super().__init__('object_recognizer')

        # -------- Configurable parameters --------
        self.declare_parameter('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        self.declare_parameter('debug_mode', False)
        self.confidence_threshold: float = self.get_parameter('confidence_threshold').value
        self.debug_mode: bool = self.get_parameter('debug_mode').value
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # -------- Device Setup --------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -------- Model (loaded exactly once) --------
        self.model_path: str = MODEL_PATH
        self.model = None
        self.class_names: Dict[int, str] = CLASS_NAMES
        self.load_model()

        # -------- Preprocessing Transforms --------
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((640, 640)),
            T.ToTensor()
        ])

        # -------- ROS interface --------
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        self.publisher_sign_board = self.create_publisher(
            String,
            '/sign_board_detection',
            QOS_PROFILE_DEFAULT)

        self.publisher_debug_image = self.create_publisher(
            CompressedImage,
            '/debug_images/sign_board_detection',
            QOS_PROFILE_DEFAULT)

    def _on_parameters_changed(self, params) -> SetParametersResult:
        """Live-update confidence_threshold / debug_mode via ros2 param set."""
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.debug_mode = self.get_parameter('debug_mode').value
        return SetParametersResult(successful=True)

    # =====================================================================
    # load_model
    # =====================================================================
    def load_model(self) -> None:
        """Loads PyTorch / TorchScript model directly without ultralytics."""
        try:
            try:
                # Attempt loading TorchScript model first
                self.model = torch.jit.load(self.model_path, map_location=self.device)
            except Exception:
                # Fallback to standard PyTorch model loading
                loaded_object = torch.load(self.model_path, map_location=self.device)
                if isinstance(loaded_object, dict) and 'model' in loaded_object:
                    self.model = loaded_object['model'].to(self.device)
                else:
                    self.model = loaded_object.to(self.device)

            self.model.eval()
            self.get_logger().info('PyTorch model loaded successfully.')
        except Exception as error:
            self.get_logger().error(
                f'Failed to load PyTorch model from "{self.model_path}": {error}. '
                f'Object recognizer will remain idle until this is fixed.')
            self.model = None

    # =====================================================================
    # decode_image
    # =====================================================================
    def decode_image(self, message: CompressedImage) -> Optional[np.ndarray]:
        """Convert ROS CompressedImage to BGR numpy array."""
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as error:
            self.get_logger().error(f'Failed to decode camera image: {error}')
            return None
        return image

    # =====================================================================
    # run_inference
    # =====================================================================
    def run_inference(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Executes forward pass using PyTorch directly."""
        try:
            # Convert BGR (OpenCV) to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model(input_tensor)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
            return preds
        except Exception as error:
            self.get_logger().error(f'PyTorch inference failed: {error}')
            return None

    # =====================================================================
    # filter_detections
    # =====================================================================
    def filter_detections(self, raw_preds: torch.Tensor, orig_shape: Tuple[int, int]) -> List[Detection]:
        """Applies confidence threshold, Non-Maximum Suppression (NMS), and rescales boxes."""
        detections: List[Detection] = []
        if raw_preds is None or len(raw_preds) == 0:
            return detections

        # Process prediction tensor shape [1, num_boxes, 5 + num_classes] or [1, 5 + num_classes, num_boxes]
        preds = raw_preds[0] if raw_preds.dim() == 3 else raw_preds
        if preds.shape[0] < preds.shape[1]:
            preds = preds.transpose(0, 1)

        boxes = preds[:, :4]
        scores = preds[:, 4:]

        if scores.shape[1] > 1:
            class_conf, class_ids = torch.max(scores, dim=1)
        else:
            class_conf = scores.squeeze(1)
            class_ids = torch.zeros_like(class_conf, dtype=torch.long)

        # Filter low confidence
        mask = class_conf >= self.confidence_threshold
        boxes, class_conf, class_ids = boxes[mask], class_conf[mask], class_ids[mask]

        if boxes.shape[0] == 0:
            return detections

        # Convert center_x, center_y, w, h to x1, y1, x2, y2 if necessary
        if boxes[0, 2] < boxes[0, 0] or boxes[0, 3] < boxes[0, 1]:
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # Apply Torchvision NMS
        keep_indices = torchvision.ops.nms(boxes, class_conf, IOU_THRESHOLD)
        boxes = boxes[keep_indices]
        class_conf = class_conf[keep_indices]
        class_ids = class_ids[keep_indices]

        # Rescale bounding boxes back to original image dimensions
        orig_h, orig_w = orig_shape
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        for box, conf, cid in zip(boxes, class_conf, class_ids):
            x1 = float(box[0] * scale_x)
            y1 = float(box[1] * scale_y)
            x2 = float(box[2] * scale_x)
            y2 = float(box[3] * scale_y)
            bbox_center_x = (x1 + x2) / 2.0

            detections.append((bbox_center_x, int(cid), float(conf), [x1, y1, x2, y2]))

        return detections

    # =====================================================================
    # sort_detections
    # =====================================================================
    def sort_detections(self, detections: List[Detection]) -> List[Detection]:
        """Order detections left-to-right by bbox center x."""
        return sorted(detections, key=lambda detection: detection[0])

    # =====================================================================
    # convert_class_name
    # =====================================================================
    def convert_class_name(self, class_id: int) -> Optional[str]:
        """Convert class ID into LABEL:DIRECTION token."""
        class_name = self.class_names.get(class_id, "")

        if '_' not in class_name:
            self.get_logger().warn(
                f'Skipping detection with unexpected class name format: {class_name}')
            return None

        label, direction = class_name.split('_', 1)
        return f'{label}:{direction}'

    # =====================================================================
    # build_navigation_string
    # =====================================================================
    def build_navigation_string(self, sorted_detections: List[Detection]) -> str:
        """Assemble semicolon-delimited string."""
        tokens: List[str] = []
        for _, class_id, _, _ in sorted_detections:
            token = self.convert_class_name(class_id)
            if token is not None:
                tokens.append(token)
        return ';'.join(tokens)

    # =====================================================================
    # publish_navigation
    # =====================================================================
    def publish_navigation(self, navigation_string: str) -> None:
        """Publish result to /sign_board_detection."""
        if not navigation_string:
            return

        message = String()
        message.data = navigation_string
        self.publisher_sign_board.publish(message)
        self.get_logger().info(f'Detected Sign Board: {navigation_string}')

    # =====================================================================
    # classify_sign
    # =====================================================================
    def classify_sign(self, image: np.ndarray) -> str:
        """Pipeline orchestration."""
        raw_preds = self.run_inference(image)
        if raw_preds is None:
            return ''

        detections = self.filter_detections(raw_preds, (image.shape[0], image.shape[1]))
        if not detections:
            return ''

        sorted_detections = self.sort_detections(detections)
        navigation_string = self.build_navigation_string(sorted_detections)

        if self.debug_mode:
            self._publish_debug_image(image, sorted_detections)

        return navigation_string

    # =====================================================================
    # camera_image_callback
    # =====================================================================
    def camera_image_callback(self, message: CompressedImage) -> None:
        if self.model is None:
            return

        image = self.decode_image(message)
        if image is None:
            return

        navigation_string = self.classify_sign(image)
        self.publish_navigation(navigation_string)

    # =====================================================================
    # Debug visualization
    # =====================================================================
    def _publish_debug_image(self, image: np.ndarray, detections: List[Detection]) -> None:
        debug_image = image.copy()

        for _, class_id, conf, bbox in detections:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            class_name = self.class_names.get(class_id, f"ID_{class_id}")

            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name} {conf:.2f}'
            cv2.putText(
                debug_image, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        debug_message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', debug_image)
        debug_message.format = 'jpeg'
        debug_message.data = encoded_data.tobytes()
        self.publisher_debug_image.publish(debug_message)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
