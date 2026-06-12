#%%
import cv2
import subprocess
import numpy as np
from bewegungserkennung import MotionDetector


class WebcamMotionApp:
    def __init__(self, source=2, width=640, height=480):
        self.detector = MotionDetector()
        self.width = width
        self.height = height
        self.use_ffmpeg = isinstance(source, str) and source.startswith("udp")

        if self.use_ffmpeg:
            self.frame_size = width * height * 3
            self.ffmpeg = subprocess.Popen([
                "ffmpeg",
                "-i", source,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-vf", f"scale={width}:{height}",
                "pipe:1"
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(source)
            if isinstance(source, str):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.ffmpeg = None

    def read_frame(self):
        if self.use_ffmpeg:
            raw = self.ffmpeg.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            return True, frame.copy()
        else:
            return self.cap.read()

    def run(self):
        if not self.use_ffmpeg and not self.cap.isOpened():
            print("Fehler: Kamera konnte nicht geöffnet werden.")
            return

        print("Motion Detection gestartet – 'q' zum Beenden")

        ret, prev_frame = self.read_frame()
        if not ret:
            print("Fehler: Konnte ersten Frame nicht lesen.")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, curr_frame = self.read_frame()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            mask = self.detector.get_mask(prev_gray, curr_gray)
            self.detector.find_contour(mask)
            output = self.detector.draw_detections(image=curr_frame.copy())

            n = len(self.detector.objects)
            if n > 0:
                cv2.putText(output, f"Bewegung: {n} Objekt(e)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Motion Detection", output)
            cv2.imshow("Maske", mask)

            prev_gray = curr_gray

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release()

    def release(self):
        if self.cap:
            self.cap.release()
        if self.ffmpeg:
            self.ffmpeg.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Webcam:
    # app = WebcamMotionApp(source=2)

    # Raspberry Pi UDP-Stream:
    app = WebcamMotionApp(source="udp://10.126.247.41:8888", width=640, height=480)
    app.run()