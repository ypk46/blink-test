# pylint: disable=no-member
import cv2
import dlib
import click
from imutils import face_utils, resize
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
@click.option(
    "-f",
    "--file",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to video file to load.",
)
def detect(file: str):
    """
    Count how many times a person blink in a video file or feed.
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

    # Check if video file was not supplied
    if not file:
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(file)

    while True:
        # Grap current frame
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        # Resize frame, convert it to grayscale and clone original frame
        frame = resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clone = frame.copy()

        # Detect faces
        faces = detector(gray, 0)

        # Loop over the faces bounding boxes
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

            cv2.putText(
                clone,
                f"Blinks: {blinks}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Face", clone)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()  # pylint: disable=no-value-for-parameter
