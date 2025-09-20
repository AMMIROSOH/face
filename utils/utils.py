import cv2 as cv
import time

def draw_fps(frame, fps: float, time_prev: time):
    smoothing = 0.9  # EMA smoothing factor (closer to 1 = smoother)
    now = time.time()
    dt = now - time_prev
    time_prev = now
    curr_fps = 1.0 / dt if dt > 0 else 0.0
    fps = smoothing * fps + (1 - smoothing) * curr_fps  

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    alpha_box = 0.6  # box transparency (0.0 - 1.0)

    text = f"FPS: {fps:.1f}"
    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)
    margin = 10
    x = frame.shape[1] - text_w - margin
    y = text_h + margin

    # Draw semi-transparent rectangle behind text
    overlay = frame.copy()
    rect_tl = (x - 6, y - text_h - 6)
    rect_br = (x + text_w + 6, y + 6)
    cv.rectangle(overlay, rect_tl, rect_br, (0, 0, 0), -1)
    cv.addWeighted(overlay, alpha_box, frame, 1 - alpha_box, 0, frame)

    # Put the FPS text
    cv.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    return fps, time_prev