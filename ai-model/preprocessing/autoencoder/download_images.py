from pathlib import Path
import shutil
import fiftyone.zoo as foz


# Output folder
output_dir = Path("dataset/autoencoder/train")
output_dir.mkdir(parents=True, exist_ok=True)


print("Downloading COCO images...")


dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person"],
    only_matching=True,
    max_samples=100,
    shuffle=True,
    seed=42,
    dataset_name="smart_tailor_coco_large"
)


# Clear old images
for file in output_dir.glob("*"):
    if file.is_file():
        file.unlink()


count = 0

for sample in dataset:

    # Find person detections
    person_detections = []

    for detection in sample.ground_truth.detections:

        if detection.label == "person":

            person_detections.append(detection)


    if not person_detections:
        continue


    # Find largest person in image
    largest_person = max(
        person_detections,
        key=lambda d: d.bounding_box[2] * d.bounding_box[3]
    )


    x, y, width, height = largest_person.bounding_box

    person_area = width * height


    # Keep images where person occupies
    # at least 15% of the image
    if person_area < 0.15:
        continue


    source = Path(sample.filepath)

    if source.exists():

        destination = (
            output_dir /
            f"person_{count + 1:03d}{source.suffix.lower()}"
        )

        shutil.copy2(source, destination)

        count += 1


    if count >= 50:
        break


print("\n================================")
print(f"Selected images: {count}")
print("================================")

print("\nImages saved to:")
print(output_dir.resolve())