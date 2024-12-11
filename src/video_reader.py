import cv2


class VideoReader:
    """
    Video reader with buffer of the next N frames.
    Buffer is list of frames, with the first index being the oldest.
    """

    def __init__(self, file, buffer_size):
        self.video = cv2.VideoCapture(file)
        self.buffer_size = buffer_size

        self.buffer = []
        for i in range(buffer_size):
            ret, frame = self.video.read()
            assert ret
            self.buffer.append(frame)

    def __getitem__(self, index):
        return self.buffer[index]

    def next(self):
        """
        Read next frame and add to end of buffer.
        Return this frame.
        """
        ret, frame = self.video.read()
        if not ret:
            raise ValueError("No more frames to read")
        self.buffer.pop(0)
        self.buffer.append(frame)
        return frame
