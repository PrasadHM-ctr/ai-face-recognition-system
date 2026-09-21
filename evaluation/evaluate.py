import cv2
import numpy as np
import os
from itertools import combinations
from insightface.app import FaceAnalysis


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "dataset"

PEOPLE = [
    "modi",
    "virat"
]

UNKNOWN_FOLDER = "dataset/unknown"


# ==========================================
# Load model
# ==========================================

print("Loading model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Model loaded!\n")


# ==========================================
# Get embedding
# ==========================================

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", image_path)
        return None

    faces = app.get(image)

    if len(faces) != 1:

        print(
            f"Skipping {image_path} - "
            f"expected 1 face, found {len(faces)}"
        )

        return None

    embedding = faces[0].embedding

    embedding = (
        embedding /
        np.linalg.norm(embedding)
    )

    return embedding


# ==========================================
# Cosine similarity
# ==========================================

def cosine_similarity(a, b):

    return float(np.dot(a, b))


# ==========================================
# Load images for each person
# ==========================================

person_embeddings = {}


for person in PEOPLE:

    folder = os.path.join(
        DATASET_PATH,
        person
    )

    person_embeddings[person] = []

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        embedding = get_embedding(path)

        if embedding is not None:

            person_embeddings[person].append(
                (filename, embedding)
            )


# ==========================================
# GENUINE COMPARISONS
# Same person vs same person
# ==========================================

genuine_scores = []


print("\n" + "=" * 60)
print("GENUINE COMPARISONS")
print("=" * 60)


for person in PEOPLE:

    images = person_embeddings[person]

    for (name1, emb1), (name2, emb2) in combinations(
        images,
        2
    ):

        score = cosine_similarity(
            emb1,
            emb2
        )

        genuine_scores.append(score)

        print(
            f"{person}: "
            f"{name1} vs {name2} "
            f"= {score:.4f}"
        )


# ==========================================
# IMPOSTOR COMPARISONS
# Different enrolled people
# ==========================================

impostor_scores = []


print("\n" + "=" * 60)
print("IMPOSTOR COMPARISONS")
print("=" * 60)


for person1, person2 in combinations(
    PEOPLE,
    2
):

    for name1, emb1 in person_embeddings[person1]:

        for name2, emb2 in person_embeddings[person2]:

            score = cosine_similarity(
                emb1,
                emb2
            )

            impostor_scores.append(score)

            print(
                f"{person1} vs {person2}: "
                f"{name1} vs {name2} "
                f"= {score:.4f}"
            )


# ==========================================
# UNKNOWN PERSON COMPARISONS
# ==========================================

if os.path.exists(UNKNOWN_FOLDER):

    for filename in os.listdir(
        UNKNOWN_FOLDER
    ):

        path = os.path.join(
            UNKNOWN_FOLDER,
            filename
        )

        unknown_embedding = get_embedding(
            path
        )

        if unknown_embedding is None:
            continue


        for person in PEOPLE:

            for name, emb in person_embeddings[person]:

                score = cosine_similarity(
                    unknown_embedding,
                    emb
                )

                impostor_scores.append(score)

                print(
                    f"unknown vs {person}: "
                    f"{filename} vs {name} "
                    f"= {score:.4f}"
                )


# ==========================================
# Statistics
# ==========================================

print("\n" + "=" * 60)
print("SIMILARITY STATISTICS")
print("=" * 60)


print(
    "\nGenuine comparisons:",
    len(genuine_scores)
)

print(
    "Average genuine:",
    f"{np.mean(genuine_scores):.4f}"
)

print(
    "Minimum genuine:",
    f"{np.min(genuine_scores):.4f}"
)

print(
    "Maximum genuine:",
    f"{np.max(genuine_scores):.4f}"
)


print(
    "\nImpostor comparisons:",
    len(impostor_scores)
)

print(
    "Average impostor:",
    f"{np.mean(impostor_scores):.4f}"
)

print(
    "Minimum impostor:",
    f"{np.min(impostor_scores):.4f}"
)

print(
    "Maximum impostor:",
    f"{np.max(impostor_scores):.4f}"
)


# ==========================================
# Threshold analysis
# ==========================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)


thresholds = np.arange(
    0.0,
    1.01,
    0.05
)


for threshold in thresholds:

    true_accepts = sum(
        score >= threshold
        for score in genuine_scores
    )

    false_rejects = sum(
        score < threshold
        for score in genuine_scores
    )

    false_accepts = sum(
        score >= threshold
        for score in impostor_scores
    )

    true_rejects = sum(
        score < threshold
        for score in impostor_scores
    )

    total = (
        true_accepts
        + false_rejects
        + false_accepts
        + true_rejects
    )

    accuracy = (
        true_accepts + true_rejects
    ) / total


    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.2%} | "
        f"False Accept: {false_accepts} | "
        f"False Reject: {false_rejects}"
    )