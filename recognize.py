import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis


# -----------------------------------
# Configuration
# -----------------------------------

THRESHOLD = 0.30

DATABASE_PATH = "face_database/face_database.pkl"


# -----------------------------------
# Load model
# -----------------------------------

print("Loading face recognition model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Model loaded successfully!")


# -----------------------------------
# Load database
# -----------------------------------

with open(DATABASE_PATH, "rb") as file:

    database = pickle.load(file)


print("Database loaded!")

print(
    "Enrolled people:",
    list(database.keys())
)


# -----------------------------------
# Get image
# -----------------------------------

image_path = input(
    "\nEnter image path to identify: "
)

image = cv2.imread(image_path)


if image is None:

    print("ERROR: Image not found!")

    exit()


# -----------------------------------
# Detect faces
# -----------------------------------

faces = app.get(image)

print(
    "Faces detected:",
    len(faces)
)


if len(faces) == 0:

    print("No face detected.")

    exit()


# -----------------------------------
# Process each detected face
# -----------------------------------

for i, face in enumerate(faces):

    print(
        f"\n========== FACE {i + 1} =========="
    )


    # Get new face embedding

    new_embedding = face.embedding

    new_embedding = (
        new_embedding /
        np.linalg.norm(new_embedding)
    )


    best_name = None

    best_similarity = -1


    # -----------------------------------
    # Compare against database
    # -----------------------------------

    for name, embeddings in database.items():

        for stored_embedding in embeddings:

            stored_embedding = (
                stored_embedding /
                np.linalg.norm(stored_embedding)
            )


            # Cosine similarity

            similarity = float(
                np.dot(
                    new_embedding,
                    stored_embedding
                )
            )


            # Keep best match

            if similarity > best_similarity:

                best_similarity = similarity

                best_name = name


    # -----------------------------------
    # Print matching information
    # -----------------------------------

    print(
        "Best match:",
        best_name
    )

    print(
        "Similarity:",
        round(best_similarity, 4)
    )

    print(
        "Threshold:",
        THRESHOLD
    )


    # -----------------------------------
    # Unknown rejection
    # -----------------------------------

    if best_similarity >= THRESHOLD:

        print(
            "Result: KNOWN"
        )

        print(
            "Person:",
            best_name
        )

    else:

        print(
            "Result: UNKNOWN"
        )