# camera.py

import cv2
import time


class Camera:
    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720,
        show_preview=True,
        window_name="Camera Preview",
    ):
        self.show_preview = show_preview
        self.window_name = window_name

        self.cap = cv2.VideoCapture(
            camera_index
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Failed to open camera."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )


    # ---------------------------------------------------------
    # Capture
    # ---------------------------------------------------------

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None, None

        return frame, time.time()


    # ---------------------------------------------------------
    # Preview with OCR boxes
    # ---------------------------------------------------------

    def preview(
        self,
        frame,
        detections=None,
    ):

        if (
            not self.show_preview
            or frame is None
        ):
            return


        display = frame.copy()


        if detections:

            for det in detections:

                box = det.get(
                    "box"
                )

                text = det.get(
                    "text",
                    ""
                )


                if box is None:
                    continue


                pts = box.astype(int)


                cv2.polylines(
                    display,
                    [pts],
                    True,
                    (0, 255, 0),
                    2,
                )


                x = pts[0][0]
                y = pts[0][1]


                cv2.putText(
                    display,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )


        cv2.putText(
            display,
            "ENTER = capture | Q = quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
        )


        cv2.imshow(
            self.window_name,
            display
        )


    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------

    def read_key(self):

        if not self.show_preview:
            return None


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):
            return "quit"


        if key in [
            13,
            32
        ]:
            # enter or space
            return "capture"


        return None


    # ---------------------------------------------------------

    def release(self):

        if self.cap:
            self.cap.release()


        if self.show_preview:
            cv2.destroyAllWindows()