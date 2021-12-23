from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()
prediction = ImagePrediction()

def loadModel(model_path):
    
    print("Loading neural network..")
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(model_path)
    prediction.loadModel()
    print("Loading complete!")

def predict(image_path):
    predictions, probabilities = prediction.predictImage(image_path, result_count=5 )
    print("Analyzing: ", image_path)
    print()
    allPredictions = zip(predictions, probabilities)
    for eachPrediction, eachProbability in allPredictions:
        print(eachPrediction , " : " , eachProbability)

    result = "\nFor me it's a " + predictions[0] + " (" + str(round(float(probabilities[0]),2)) + "% affidability)" 
    print(result)
    return (predictions[0], str(round(float(probabilities[0]),2)))

model_dir = "models"
model_name = "resnet50_weights_tf_dim_ordering_tf_kernels.h5"
#model_name1 = "densenet201_weights_tf_dim_ordering_tf_kernels.h5"

model_path = os.path.join(execution_path, model_dir, model_name)
print("Loading model from ", model_path)
loadModel(model_path)

image_dir = "test"
image_name = "lago.jpg"
image_path = os.path.join(execution_path, image_dir, image_name)

print(predict(image_path))
