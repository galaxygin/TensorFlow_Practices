import fashion_mnist
import horse_or_human
import image_classification
import scissors_paper_rock

models = ['Scissors Paper Rock', 'Horse or Human', 'Fashion mnist', 'Fashion classify']


def train_model(selection):
    if selection == "0":
        scissors_paper_rock.conduct_training()
    if selection == "1":
        horse_or_human.conduct_training()
    if selection == "2":
        fashion_mnist.conduct_training()
    if selection == "3":
        image_classification.conduct_training()
    print("Done")
    menu()


def make_prediction(selection):
    if selection == "0":
        scissors_paper_rock.predict()
    if selection == "1":
        horse_or_human.predict()
    if selection == "2":
        fashion_mnist.predict()
    if selection == "3":
        image_classification.predict()
    print("Done")
    menu()


def menu():
    selection = input('What would you like to do?\n1: Train model\n2: Predict with model\n3: Exit\n')
    if selection == "1":
        for i in range(len(models)):
            print(repr(i)+": "+models[i])
        train_select = input("Which model would you like to train?")
        train_model(selection=train_select)
    if selection == "2":
        for i in range(len(models)):
            print(repr(i)+": "+models[i])
        predict_select = input("Which model would you like to use for prediction?")
        make_prediction(selection=predict_select)


menu()
