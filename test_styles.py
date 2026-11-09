from structured_elements_integrated import StructuredDocument

# Load the actual extracted PDF elements
doc = StructuredDocument.load_json(
    "out_sample_purchase_order/structured_elements_integrated.json"
)

# Change the shared H1 style
doc.update_type_style(
    "h1",
    font_size=30,
    color="#0000FF",
    bold=True
)

# Save the updated document
doc.save_json(
    "out_sample_purchase_order/structured_elements.json"
)

print("H1 style updated successfully!")

# Display all H1 elements
for element in doc.elements: 
    if element.type == "h1":
        print(element.content, doc.get_effective_style(element))