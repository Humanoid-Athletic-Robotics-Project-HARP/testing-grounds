"""Live camera pose estimation to K1 policy command pipeline.

Captures human pose from a webcam using MediaPipe, retargets to K1 joint
targets, and writes commands that can be consumed by the inference environment
or a real robot controller.

Architecture:
    Camera -> MediaPipe Holistic -> 3D Keypoints -> IK Retarget -> K1 Joint Angles
                                                                     |
                                                            Student Policy Input
                                                            (head + hand pos)

Requirements:
    pip install mediapipe opencv-python numpy

Usage:
    python scripts/live_camera_k1.py --mode oh2o
    python scripts/live_camera_k1.py --mode joint_angles --output_socket tcp://localhost:5555
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("Please install mediapipe: pip install mediapipe")


# K1 joint ordering (matches env config)
K1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# MediaPipe landmark indices for key body parts
MP_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class PoseEstimator:
    """Wraps MediaPipe Pose to provide 3D keypoints from webcam frames."""

    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process(self, frame_bgr):
        """Process a BGR frame and return 3D landmarks if detected.

        Returns:
            landmarks_3d: np.ndarray of shape (33, 4) with x, y, z, visibility
                          in MediaPipe normalized coords, or None if no pose detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_world_landmarks is None:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_world_landmarks.landmark],
            dtype=np.float32,
        )
        return landmarks

    def draw(self, frame_bgr, landmarks_3d=None):
        """Draw pose landmarks on the frame for visualization."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        return frame_bgr


def landmarks_to_keypoints(landmarks_3d):
    """Extract key 3D positions from MediaPipe landmarks.

    Returns dict with positions in meters (MediaPipe uses meters for world landmarks).
    Coordinate convention: x=right, y=down, z=forward (MediaPipe world).
    We convert to: x=forward, y=left, z=up (robot convention).
    """
    if landmarks_3d is None:
        return None

    def get_pos(name):
        idx = MP_LANDMARKS[name]
        mp_pos = landmarks_3d[idx, :3]
        # Convert MediaPipe world coords to robot coords
        return np.array([mp_pos[2], -mp_pos[0], -mp_pos[1]], dtype=np.float32)

    return {name: get_pos(name) for name in MP_LANDMARKS}


def compute_oh2o_commands(keypoints, k1_height=0.95):
    """Compute OmniH2O-style commands: head position + left/right hand positions.

    These are the 3D target positions that the trained student policy (OH2O mode)
    expects as its tracking command input.

    Args:
        keypoints: Dict of 3D positions from landmarks_to_keypoints.
        k1_height: Approximate K1 standing height for scaling.

    Returns:
        Dict with head_pos, left_hand_pos, right_hand_pos as np.ndarray(3,).
    """
    if keypoints is None:
        return None

    # Estimate human height from hip-to-head distance
    mid_hip = (keypoints["left_hip"] + keypoints["right_hip"]) / 2
    head = keypoints["nose"]
    human_torso_height = np.linalg.norm(head - mid_hip)

    # Scale factor: K1 torso is roughly 0.3m (trunk to head)
    k1_torso_height = 0.30
    scale = k1_torso_height / max(human_torso_height, 0.1)

    # Re-center at mid-hip
    center = mid_hip.copy()

    def scale_point(pt):
        return (pt - center) * scale

    return {
        "head_pos": scale_point(keypoints["nose"]),
        "left_hand_pos": scale_point(keypoints["left_wrist"]),
        "right_hand_pos": scale_point(keypoints["right_wrist"]),
    }


def compute_joint_angles(keypoints):
    """Compute approximate K1 joint angles from 3D keypoints using geometric IK.

    This is a simplified inverse kinematics that maps MediaPipe skeleton angles
    to K1 joint angles. For production use, a proper IK solver or learned
    retargeting model should be used.

    Returns:
        np.ndarray of shape (22,) with K1 joint angles in radians.
    """
    if keypoints is None:
        return None

    angles = np.zeros(22, dtype=np.float32)

    # Head tracking
    mid_shoulder = (keypoints["left_shoulder"] + keypoints["right_shoulder"]) / 2
    head_dir = keypoints["nose"] - mid_shoulder
    if np.linalg.norm(head_dir[:2]) > 1e-6:
        angles[0] = np.arctan2(head_dir[1], head_dir[0])  # AAHead_yaw
    angles[1] = np.arctan2(-head_dir[2], np.linalg.norm(head_dir[:2]))  # Head_pitch

    def arm_angles(shoulder, elbow, wrist, side="left"):
        """Compute shoulder and elbow angles for one arm."""
        upper_arm = elbow - shoulder
        forearm = wrist - elbow

        upper_arm_len = max(np.linalg.norm(upper_arm), 1e-6)
        upper_arm_dir = upper_arm / upper_arm_len

        # Shoulder pitch (forward/backward rotation)
        s_pitch = np.arctan2(-upper_arm_dir[2], upper_arm_dir[0])
        # Shoulder roll (lateral raise)
        s_roll = np.arcsin(np.clip(upper_arm_dir[1] if side == "left" else -upper_arm_dir[1], -1, 1))

        # Elbow angle
        forearm_len = max(np.linalg.norm(forearm), 1e-6)
        forearm_dir = forearm / forearm_len
        cos_elbow = np.clip(np.dot(upper_arm_dir, forearm_dir), -1, 1)
        e_pitch = np.pi - np.arccos(cos_elbow)

        return s_pitch, s_roll, e_pitch

    # Left arm
    lsp, lsr, lep = arm_angles(
        keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"], "left"
    )
    angles[2] = np.clip(lsp, -3.316, 1.22)   # ALeft_Shoulder_Pitch
    angles[3] = np.clip(lsr, -1.74, 1.57)    # Left_Shoulder_Roll
    angles[4] = np.clip(lep, -2.27, 2.27)    # Left_Elbow_Pitch

    # Right arm
    rsp, rsr, rep = arm_angles(
        keypoints["right_shoulder"], keypoints["right_elbow"], keypoints["right_wrist"], "right"
    )
    angles[6] = np.clip(rsp, -3.316, 1.22)   # ARight_Shoulder_Pitch
    angles[7] = np.clip(rsr, -1.57, 1.74)    # Right_Shoulder_Roll
    angles[8] = np.clip(rep, -2.27, 2.27)    # Right_Elbow_Pitch

    return angles


class CommandPublisher:
    """Publishes retargeted commands via file, socket, or stdout."""

    def __init__(self, output_method="file", output_path="k1_live_commands.csv"):
        self.method = output_method
        self.path = output_path
        self._socket = None

        if output_method == "file":
            self._file = open(output_path, "w")
            header = "timestamp," + ",".join(K1_JOINT_NAMES)
            self._file.write(header + "\n")
        elif output_method == "socket":
            try:
                import zmq
                context = zmq.Context()
                self._socket = context.socket(zmq.PUB)
                self._socket.bind(output_path)
                print(f"Publishing commands on {output_path}")
            except ImportError:
                raise ImportError("ZMQ output requires pyzmq: pip install pyzmq")

    def publish(self, timestamp, joint_angles=None, oh2o_commands=None):
        if joint_angles is not None:
            if self.method == "file":
                line = f"{timestamp:.4f}," + ",".join(f"{a:.6f}" for a in joint_angles)
                self._file.write(line + "\n")
                self._file.flush()
            elif self.method == "socket" and self._socket:
                import json
                msg = {
                    "timestamp": timestamp,
                    "joint_angles": joint_angles.tolist(),
                }
                if oh2o_commands:
                    msg["oh2o"] = {k: v.tolist() for k, v in oh2o_commands.items()}
                self._socket.send_json(msg)
            elif self.method == "stdout":
                print(f"t={timestamp:.3f} angles={joint_angles[:6]}...")

    def close(self):
        if hasattr(self, "_file") and self._file:
            self._file.close()


def main():
    parser = argparse.ArgumentParser(description="Live camera to K1 pose commands")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--mode", choices=["oh2o", "joint_angles", "both"], default="both",
                        help="Command output mode")
    parser.add_argument("--output", type=str, default="stdout",
                        help="Output method: 'stdout', 'file:path.csv', or 'socket:tcp://...'")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS")
    parser.add_argument("--show", action="store_true", default=True, help="Show webcam preview")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()

    # Set up output
    if args.output.startswith("file:"):
        publisher = CommandPublisher("file", args.output[5:])
    elif args.output.startswith("socket:"):
        publisher = CommandPublisher("socket", args.output[7:])
    else:
        publisher = CommandPublisher("stdout")

    # Set up camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FPS, args.fps)
    print(f"Camera opened at {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

    estimator = PoseEstimator()

    # Smoothing filter for output
    angle_history = deque(maxlen=3)
    frame_time = 1.0 / args.fps

    print("Starting live capture. Press 'q' to quit.")
    start_time = time.time()

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Run pose estimation
            landmarks = estimator.process(frame)
            keypoints = landmarks_to_keypoints(landmarks)

            timestamp = time.time() - start_time

            if keypoints is not None:
                oh2o_cmd = None
                joint_angles = None

                if args.mode in ("oh2o", "both"):
                    oh2o_cmd = compute_oh2o_commands(keypoints)

                if args.mode in ("joint_angles", "both"):
                    raw_angles = compute_joint_angles(keypoints)
                    if raw_angles is not None:
                        angle_history.append(raw_angles)
                        joint_angles = np.mean(angle_history, axis=0)

                publisher.publish(timestamp, joint_angles=joint_angles, oh2o_commands=oh2o_cmd)

            if args.show:
                frame = estimator.draw(frame)
                cv2.putText(frame, f"FPS: {1.0 / max(time.time() - t0, 0.001):.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if keypoints is not None:
                    cv2.putText(frame, "TRACKING", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO POSE", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("K1 Live Camera", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Rate limiting
            elapsed = time.time() - t0
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nStopping capture.")

    cap.release()
    cv2.destroyAllWindows()
    publisher.close()
    print("Done.")


if __name__ == "__main__":
    main()
