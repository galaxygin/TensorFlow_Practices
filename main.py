import airplane_classifier
import cat_or_dog
import fashion_classifier
import flower_classifier
import horse_or_human
import fashion_mnist
import number_rule_prediction
import scissors_paper_rock
import sex_classifier

models = ['1: Rule of numbers', '2: Fashion mnist', '3: Fashion', '4: Scissors Paper Rock', '5: Horse or Human',
          '6: Flowers', '7: Cat or Dog', '8: Airplanes']
trained_num_rule = number_rule_prediction.TrainedModel()
trained_fashion_mnist = fashion_mnist.TrainedModel()
trained_fashion = fashion_classifier.TrainedModel()
trained_rps = scissors_paper_rock.TrainedModel()
trained_horse_or_human = horse_or_human.TrainedModel()
trained_flowers = flower_classifier.TrainedModel()
trained_cat_or_dog = cat_or_dog.TrainedModel()
trained_airplanes = airplane_classifier.TrainedModel()
trained_sex = sex_classifier.TrainedModel()


def train_model(selection):
    if selection == "1":
        number_rule_prediction.conduct_training()
    if selection == "2":
        fashion_mnist.conduct_training()
    if selection == "3":
        fashion_classifier.conduct_training()
    if selection == "4":
        scissors_paper_rock.conduct_training()
    if selection == "5":
        horse_or_human.conduct_training()
    if selection == "6":
        flower_classifier.conduct_training()
    if selection == "7":
        cat_or_dog.conduct_training()
    if selection == "8":
        airplane_classifier.conduct_training()
    print("Done")
    menu()


def make_prediction(selection):
    if selection == "1":
        trained_num_rule.predict()
    if selection == "2":
        trained_fashion_mnist.predict()
    if selection == "3":
        trained_fashion.predict()
    if selection == "4":
        trained_rps.predict()
    if selection == "5":
        trained_horse_or_human.predict()
    if selection == "6":
        trained_flowers.predict()
    if selection == "7":
        trained_cat_or_dog.predict()
    if selection == "8":
        trained_airplanes.predict()
    print("Done")
    menu()


def model_summary(selection):
    if selection == "1":
        trained_num_rule.summary()
    if selection == "2":
        trained_fashion_mnist.evaluate()
    if selection == "3":
        trained_fashion.summary()
    if selection == "4":
        trained_rps.summary()
    if selection == "5":
        trained_horse_or_human.summary()
    if selection == "6":
        trained_flowers.summary()
    if selection == "7":
        trained_cat_or_dog.summary()
    if selection == "8":
        trained_airplanes.summary()
    menu()


def menu():
    selection = input('What would you like to do?\n1: Train model\n2: Predict with model\n3: Model summary\n4: Exit\n')
    if selection == "1":
        for name in models:
            print(name)
        train_select = input("Which model would you like to train?")
        train_model(selection=train_select)
    if selection == "2":
        for name in models:
            print(name)
        predict_select = input("Which model would you like to use for prediction?")
        make_prediction(selection=predict_select)
    if selection == "3":
        for name in models:
            print(name)
        summary_select = input("Which summary of model would you like to see?")
        model_summary(selection=summary_select)


menu()
