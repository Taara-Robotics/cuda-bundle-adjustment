import cuba
import json
import time
import numpy as np
import sys

ba_input_path = sys.argv[1]

with open(ba_input_path) as f:
    data = json.load(f)

# Load camera intrinsics
intrinsics = (
    data["fx"],
    data["fy"],
    data["cx"],
    data["cy"],
)

# Load camera poses
poses_t = []
poses_q = []
poses_fixed = []
pose_indices = {}

for i, vertex in enumerate(data["pose_vertices"]):
    poses_t.append(vertex["t"])
    poses_q.append(vertex["q"])
    pose_indices[vertex["id"]] = i
    poses_fixed.append(vertex["fixed"])

poses_t = np.array(poses_t)
poses_q = np.array(poses_q)

# Load landmarks
landmarks = []
landmark_indices = {}

for i, landmark in enumerate(data["landmark_vertices"]):
    landmarks.append(landmark["Xw"])
    landmark_indices[landmark["id"]] = i

landmarks = np.array(landmarks)

# Load observations
edges = []
measurements = []

for edge in data["monocular_edges"]:
    edges.append((pose_indices[edge["vertexP"]], landmark_indices[edge["vertexL"]]))
    measurements.append(edge["measurement"])

edges = np.array(edges)
measurements = np.array(measurements)

# Run BA
start_time = time.time()
optimized_poses_t, optimized_poses_q, optimized_landmarks, edge_chi_squares  = cuba.run_ba(intrinsics, poses_t, poses_q, landmarks, edges, measurements, 10)

print("BA time:", time.time() - start_time, "s")
print(f"Poses: {len(optimized_poses_t)}, landmarks: {len(optimized_landmarks)}, edges: {len(edges)}")
print("Sum of chi2:", np.sum(edge_chi_squares))

