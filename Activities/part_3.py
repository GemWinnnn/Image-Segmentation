#Model Evaluation
scores = model.evaluate(X, y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss and accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy']) 
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Prediction
import seaborn as sns
from sklearn.netrics import confusion_natrix
def plt_show(ing): 
    plt.imshow(img) 
    plt.show()

cup   = 'img_1/10.png'
spoon = 'img_2/10.png'
fork  = 'img_3/10.png'
mouse = 'img_4/10.png'

imgs  = [cup, spoon, fork, mouse]

# def predict (img_path):
classes = None
predicted_classes = []

for i in range(len( imgs)):
    type_      = preprocessing.image.load_img(imgs[i], target_size=(width, height)) 
    plt.imshow(type_)
    plt.show()

    type_x     = np.expand_dims (type_, axis=0)
    prediction = model.predict(type_x)
    index      = np.argmax(prediction)
    print(class_names[index])
    classes    = class_names[index] 
    predicted_classes.append(class_names[index])

cm = confusion_matrix(class_names, predicted_classes)
f = sns.heatmap(cm, xticklabels=class_names, yticklabels=predicted_classes, annot=True)

type_1 = preprocessing.image.load_img('img_1/10.png', target_size=(width, height))

plt.imshow(type_1)
plt.show()

type_1_x = np.expand_dims (type_1, axis=0)
predictions = model.predict(type_1_x)
index = np.argmax (predictions)

print(class_names[index])

type_2 = preprocessing.image.load_img('img_2/10.png', target_size=(width, height))

plt.imshow(type_2)
plt.show()

type_2_x = np.expand_dims (type_2, axis=0)

predictions = model.predict(type_2_x)

index = np.argmax (predictions) print(class_names[index])


