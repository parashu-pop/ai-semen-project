from ultralytics import YOLO
import cv2
import math
from collections import defaultdict

# =========================
# LOAD YOLO MODEL
# =========================

# =========================
# LOAD NEW YOLO MODEL
# =========================
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "dg_exp1"
    / "weights"
    / "last.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"New YOLO model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print(f"NEW YOLO MODEL LOADED: {MODEL_PATH}")

# =========================
# VIDEO PATH
# =========================
video_path = "temp_video.mp4"

cap = cv2.VideoCapture(video_path)

# =========================
# VIDEO INFO
# =========================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# =========================
# OUTPUT VIDEO
# =========================
out = cv2.VideoWriter(
    "outputs/sperm_tracking_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# =========================
# TRACK HISTORY
# =========================
track_history = defaultdict(list)


# =========================
# FINAL MOTILITY STORAGE
# =========================
final_motility = {}

# =========================
# FRAME LOOP
# =========================
while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # =========================
    # YOLO + BYTE TRACK
    # =========================
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.5,
        verbose=False
    )

    annotated_frame = frame.copy()

    current_ids = set()

    # =========================
    # CHECK DETECTIONS
    # =========================
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        current_ids = set(track_ids)


        for box, track_id, conf in zip(
            boxes,
            track_ids,
            confidences
        ):

            x, y, w, h = box

            # =========================
            # BOX COORDINATES
            # =========================
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            center = (float(x), float(y))

            # =========================
            # STORE TRACK HISTORY
            # =========================
            track = track_history[track_id]
            track.append(center)

            # KEEP ONLY LAST 15 POSITIONS
            if len(track) > 15:
                track.pop(0)

            # =========================
            # MOTILITY ANALYSIS
            # =========================
            motion_distance = 0

            if len(track) >= 2:

                start_x, start_y = track[0]
                end_x, end_y = track[-1]

                motion_distance = math.sqrt(
                    (end_x - start_x) ** 2 +
                    (end_y - start_y) ** 2
                )
                print(
                f"ID {track_id} -> Distance = {motion_distance:.2f}"
                 )


                if motion_distance > 6:

                    motility = "Progressive"
                    color = (0, 255, 0)

                elif motion_distance > 2:

                    motility = "Non-Progressive"
                    color = (0, 165, 255)

                else:

                    motility = "Immotile"
                    color = (0, 0, 255)

                if track_id not in final_motility:

                    final_motility[track_id] = motility

                else:

                    previous = final_motility[track_id]

                    if previous == "Immotile" and motility != "Immotile":

                        final_motility[track_id] = motility

                    elif previous == "Non-Progressive" and motility == "Progressive":

                        final_motility[track_id] = motility


            else:

                motility = "Tracking"
                color = (255, 255, 0)
            # =========================
            # DRAW BOX
            # =========================
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # =========================
            # LABEL
            # =========================
            label = f"ID:{track_id} {conf:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # =========================
            # TRAJECTORY
            # =========================
            points = track_history[track_id]

            for i in range(1, len(points)):

                pt1 = (
                    int(points[i - 1][0]),
                    int(points[i - 1][1])
                )

                pt2 = (
                    int(points[i][0]),
                    int(points[i][1])
                )

                cv2.line(
                    annotated_frame,
                    pt1,
                    pt2,
                    (255, 0, 0),
                    2
                )

    # =========================
    # REAL CURRENT TOTAL
    # =========================
    total_sperm = len(current_ids)

    # =========================
    # TRANSPARENT PANEL
    # =========================
    overlay = annotated_frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (450, 190),
        (0, 0, 0),
        -1
    )

    alpha = 0.6

    cv2.addWeighted(
        overlay,
        alpha,
        annotated_frame,
        1 - alpha,
        0,
        annotated_frame
    )
    progressive_display = sum(
    1 for v in final_motility.values()
    if v == "Progressive"
     )

    non_progressive_display = sum(
    1 for v in final_motility.values()
    if v == "Non-Progressive"
     )

    immotile_display = sum(
    1 for v in final_motility.values()
    if v == "Immotile"
    )

    display_total = len(final_motility)

    # =========================
    # DISPLAY TEXT
    # =========================

    cv2.putText(
        annotated_frame,
       f"Total Sperm: {display_total}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
       f"Progressive: {progressive_display}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
       f"Non-Progressive: {non_progressive_display}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 165, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Immotile: {immotile_display}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # =========================
    # SHOW VIDEO
    # =========================
    cv2.imshow("Sperm Tracking", annotated_frame)

    # =========================
    # SAVE OUTPUT
    # =========================
    out.write(annotated_frame)

    # =========================
    # EXIT KEY
    # =========================
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# RELEASE EVERYTHING
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

# =========================
# FINAL RESULTS
# =========================
progressive_count = sum(
    1 for v in final_motility.values()
    if v == "Progressive"
)

non_progressive_count = sum(
    1 for v in final_motility.values()
    if v == "Non-Progressive"
)

immotile_count = sum(
    1 for v in final_motility.values()
    if v == "Immotile"
)

total_sperm = len(final_motility)

print("\n========== FINAL RESULTS ==========")
print(f"Final Visible Sperm Count: {total_sperm}")
print(f"Progressive: {progressive_count}")
print(f"Non-Progressive: {non_progressive_count}")
print(f"Immotile: {immotile_count}")

with open(
    "outputs/final_tracking_results.txt",
    "w"
) as f:

    f.write(
        f"Final Visible Sperm Count: {total_sperm}\n"
    )

    f.write(
        f"Progressive: {progressive_count}\n"
    )

    f.write(
        f"Non-Progressive: {non_progressive_count}\n"
    )

    f.write(
        f"Immotile: {immotile_count}\n"
    )
