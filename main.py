import fashion_mnist
import horse_or_human
import image_classification
import number_rule_prediction
import scissors_paper_rock

models = ['1: Scissors Paper Rock', '2: Horse or Human', '3: Fashion mnist', '4: Fashion classify',
          "5: Rule of numbers"]
trained_rps = scissors_paper_rock.TrainedModel()
trained_horse_or_human = horse_or_human.TrainedModel()
trained_fashion_mnist = fashion_mnist.TrainedModel()
trained_fashion_classify = image_classification.TrainedModel()
trained_num_rule = number_rule_prediction.TrainedModel()


def train_model(selection):
    if selection == "1":
        scissors_paper_rock.conduct_training()
    if selection == "2":
        horse_or_human.conduct_training()
    if selection == "3":
        fashion_mnist.conduct_training()
    if selection == "4":
        image_classification.conduct_training()
    if selection == "5":
        number_rule_prediction.conduct_training()
    print("Done")
    menu()


def make_prediction(selection):
    if selection == "1":
        trained_rps.predict()
    if selection == "2":
        trained_horse_or_human.predict()
    if selection == "3":
        trained_fashion_mnist.predict()
    if selection == "4":
        trained_fashion_classify.predict()
    if selection == "5":
        trained_num_rule.predict()
    print("Done")
    menu()


def model_summary(selection):
    if selection == "1":
        trained_rps.summary()
    if selection == "2":
        trained_horse_or_human.summary()
    if selection == "3":
        trained_fashion_mnist.evaluate()
    if selection == "4":
        trained_fashion_classify.summary()
    if selection == "5":
        trained_num_rule.summary()
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
