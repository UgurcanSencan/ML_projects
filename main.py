from imageai.Detection import ObjectDetection

model = ObjectDetection()

path_model = "./Models/m1.h5"
path_input = "./Input/images.jpg"
path_output = "./Output/newimage.jpg"

model.setModelTypeAsYOLOv3()
model.setModelPath(path_model)
model.loadModel()
recognition = model.detectObjectsFromImage(
    input_image=path_input,
    output_image_path = path_output
)

for eachItem in recognition:
    print(eachItem["name"], ":", eachItem["percentage_probability"])

