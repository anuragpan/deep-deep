from keras.models import load_model
import cv2
import numpy as np
from random import choice
import time,sys

score1 = 0
score2 = 0
name1 = str(sys.argv[1])
name2 = str(sys.argv[2])
winner = "Waiting..."


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

prev_move1 = None
prev_move2 = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (0, 100), (300, 400), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (330, 100), (630, 400), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi1= frame[100:400, 0:300]
    img1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (227, 227))

    # predict the move made
    pred1 = model.predict(np.array([img1]))
    move_code1 = np.argmax(pred1[0])
    user_move_name1 = mapper(move_code1)

    roi2= frame[100:400, 330:630]
    img2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (227, 227))

    # predict the move made
    pred2 = model.predict(np.array([img2]))
    move_code2 = np.argmax(pred2[0])
    user_move_name2 = mapper(move_code2)

    # predict the winner (human vs computer)
    if prev_move1 != user_move_name1 or prev_move2 != user_move_name2 :
        if user_move_name1 != "none" and user_move_name2 != "none":
            winner = calculate_winner(user_move_name1, user_move_name2)
            if(winner == "User"):
                score1 +=1
                winner = name1
            elif(winner == "Computer"):
                score2 +=1
                winner = name2


        else:
            winner = "Waiting..."
    prev_move1 = user_move_name1
    prev_move2 = user_move_name2
    time.sleep(0.05)

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, name1 +": " + user_move_name1,
                (50, 50), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame,name2 + ": " + user_move_name2,
                (450, 50), font, 1.2, ( 255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + str(winner),
                (80, 450), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    # if computer_move_name != "none":
    #     icon = cv2.imread(
    #         "images/{}.png".format(computer_move_name))
    #     icon = cv2.resize(icon, (300, 300))
    #     frame[100:400, 340:640] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("\n       "+str(name1)+"\n     Your Score: "+str(score1)+"\n\n")
print("\n       "+str(name2)+"\n     Your Score: "+str(score2)+"\n\n")

