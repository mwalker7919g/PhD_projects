# classes for recording videos
import cv2
import time
import threading

class RecordVideo():

    def __init__(self, video_filename, capture_duration):

        self.open = True
        self.capture_duration = capture_duration # length of video
        self.device_index, self.fps, self.fourcc, self.frameSize = 0, 27, "MJPG", (640, 480) # video parameters
        self.video_filename = video_filename
        self.video_cap, self.video_writer = cv2.VideoCapture(self.device_index), cv2.VideoWriter_fourcc(*self.fourcc) # save video
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()

    def record(self):

        while int(time.time() - self.start_time) < self.capture_duration:
            ret, video_frame = self.video_cap.read()
            if ret:
                self.video_out.write(video_frame)
                self.frame_counts += 1
            else:
                break

    def stop(self):

        if self.open:

            self.open = False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()
        else:
            pass

    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()


def start_video_recording(video_filename, vial_size_input, capture_duration):
    global video_thread
    if vial_size_input == int(8): # length of video changes for vial size

        video_thread = RecordVideo(video_filename, capture_duration=5)
        video_thread.start()

    elif vial_size_input == int(20):

        video_thread = RecordVideo(video_filename, capture_duration=20)
        video_thread.start()

    return video_filename

def stop_video_recording(video_filename):
    global video_thread
    video_thread.stop()
