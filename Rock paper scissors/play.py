from keras.models import load_model
import cv2
import numpy as np
from random import choice
import time,sys

score = 0
name = sys.argv[1]


REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (0, 100), (300, 400), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (340, 100), (640, 400), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:400, 0:300]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
            if(winner == "User"):
                score +=1
            elif(winner == "Computer"):
                score -=1


        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name
    time.sleep(0.1)

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "" + user_move_name,
                (50, 50), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "" + computer_move_name,
                (450, 50), font, 1.2, ( 255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (80, 450), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (300, 300))
        frame[100:400, 340:640] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("\n       "+str(name)+"\n     Your Score: "+str(score)+"\n\n")
