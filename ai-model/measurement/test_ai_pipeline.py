from ai_pipeline import analyze_body


image_path = "test_person.jpg"

user_height = 165.0


result = analyze_body(
    image_path,
    user_height
)


print("\n========== AI PIPELINE RESULT ==========")

for key, value in result.items():

    print(
        f"{key}: {value}"
    )