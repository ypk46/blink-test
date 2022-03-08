import click
import dlib
import cv2 as cv
from imutils import face_utils, resize
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist


EAR_THRESHOLD = 0.19
EAR_CONTINOUS_FRAME_THRESHOLD = 3
LANDMARK_PREDICTOR_FILE = "data/predictor.dat"


def get_eye_aspect_ratio(eye: list):
    """
    Calculate the eye aspect ratio which is the relation
    between the vertial and horizontal distance of the eye
    coordinates.
    """
    vertical_a = dist.euclidean(eye[1], eye[5])
    vertical_b = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (vertical_a + vertical_b) / (2.0 * horizontal)
    return ear


@click.command()
@click.option("-f", "--file", type=click.Path(exists=True, dir_okay=False))
def count_blink(file: str):
    """
    Count how many times a person blink in a video file.
    """
    blinks = 0
    counter = 0

    # Initialize face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()  # pylint: disable=no-member
    predictor = dlib.shape_predictor(  # pylint: disable=no-member
        LANDMARK_PREDICTOR_FILE
    )

    # Get eye landmarks coordinate indexes
    (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Fetch video stream
    video = FileVideoStream(file).start()

    while True:
        # Stop when reaching the end of the video.
        if not video.more():
            break

        # Grap a frame and preprocess it
        frame = video.read()
        if frame is None:
            break

        frame = resize(frame, width=450)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # pylint: disable=no-member

        # Detect faces
        faces = detector(gray, 0)

        # Loop over the face detections
        for face in faces:
            # Get eyes landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[left_start:left_end]
            right_eye = shape[right_start:right_end]

            # Calculate eye aspect ratio of both eyes
            left_aspect_ratio = get_eye_aspect_ratio(left_eye)
            right_aspect_ratio = get_eye_aspect_ratio(right_eye)

            # Calculate the avg eye aspect ratio
            aspect_ratio = (left_aspect_ratio + right_aspect_ratio) / 2.0

            # Increase counter if EAR is below threshold
            if aspect_ratio < EAR_THRESHOLD:
                counter += 1

            else:
                # Check if eyes where closed for enough frames in order to
                # count it as a blink.
                if counter >= EAR_CONTINOUS_FRAME_THRESHOLD:
                    blinks += 1

                # Reset counter
                counter = 0

    click.echo(f"Detected {blinks} blinks in the video!")


if __name__ == "__main__":
    count_blink()  # pylint: disable=no-value-for-parameter
