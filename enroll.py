import cv2
import pickle
import os
from insightface.app import FaceAnalysis


# -----------------------------------
# Load model
# -----------------------------------

print("Loading face recognition model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("Model loaded successfully!")


# -----------------------------------
# Create database folder
# -----------------------------------

os.makedirs("face_database", exist_ok=True)

database_path = "face_database/face_database.pkl"


# -----------------------------------
# Get person information
# -----------------------------------

name = input("Enter person's name: ")

image_path = input("Enter image path: ")


# -----------------------------------
# Load image
# -----------------------------------

image = cv2.imread(image_path)

if image is None:

    print("ERROR: Image not found!")

    exit()


# -----------------------------------
# Detect faces
# -----------------------------------

faces = app.get(image)

print("Faces detected:", len(faces))


# -----------------------------------
# Validate face count
# -----------------------------------

if len(faces) == 0:

    print("ERROR: No face detected.")

    exit()


if len(faces) > 1:

    print(
        "ERROR: Multiple faces detected."
    )

    print(
        "Please use an image containing "
        "only one person."
    )

    exit()


# -----------------------------------
# Get embedding
# -----------------------------------

embedding = faces[0].embedding


# -----------------------------------
# Load existing database
# -----------------------------------

if os.path.exists(database_path):

    with open(database_path, "rb") as file:

        database = pickle.load(file)

else:

    database = {}


# -----------------------------------
# Add embedding
# -----------------------------------

if name in database:

    database[name].append(embedding)

else:

    database[name] = [embedding]


# -----------------------------------
# Save database
# -----------------------------------

with open(database_path, "wb") as file:

    pickle.dump(database, file)


print("\nEnrollment successful!")

print("Person:", name)

print(
    "Total embeddings for this person:",
    len(database[name])
)

print(
    "Database saved at:",
    database_path
)