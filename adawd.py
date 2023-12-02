import random
import tkinter as tk

from PIL import Image, ImageTk

random_deck = []  # 隨機牌堆
my_deck = []  # 自己的牌堆


def initialize_game():
    global random_deck, my_deck
    random_deck = []
    my_deck = []
    for num in range(1, 10):
        random_deck.extend([num] * 2)  # 每種牌有兩張，放入隨機牌堆
        my_deck.append(0)  # 自己的牌堆初始為0
    random.shuffle(random_deck)  # 隨機洗牌


# Modify the image loading path to match your actual image directory
# Change the path to the location where `card_10.png` exists
# It is assumed card_10.png is in the same images subfolder as the other card images
image_path = "images/card_10.png"
# 載入圖像並創建圖像物件
image = Image.open(image_path)
photo = ImageTk.PhotoImage(image)
# 創建圖像標籤
label.configure(image=photo)  # Corrected line
label.image = photo  # keep a reference!


def move_card_left():
    global my_deck
    for i in range(len(my_deck)-1):
        my_deck[i] = my_deck[i+1]  # 將牌堆中的每張牌向左移動一格
    my_deck[-1] = 0  # 最右邊的位置變成空白


def draw_card():
    global random_deck, my_deck
    if random_deck:
        card = random_deck.pop(0)  # 從隨機牌堆抽一張牌
        empty_slot = my_deck.index(0)  # 找到自己牌堆中的一個空位
        my_deck[empty_slot] = card  # 放入自己的牌堆


def show_card_images():
    def update_card_images():
        for i in range(len(my_deck)):
            labels[i].configure(image=images[my_deck[i]])

    def check_win():
        return all(card == 0 for card in my_deck)  # 檢查自己的牌堆是否全部為0

    def play_game():
        initialize_game()
        turns = 0
        while not check_win():
            turns += 1
            move_card_left()
            draw_card()
            update_card_images()
            root.update()  # 更新圖形介面
        result_label.configure(text="遊戲結束！總回合數: " + str(turns))

    def restart_program():
        # 在這裡編寫重新開始程式的相關邏輯
        print("程式重新開始")

    root = tk.Tk()
    root.title("牌堆圖片展示")

    canvas = tk.Canvas(root, width=400, height=200)
    canvas.pack()

    # 設置背景圖片
    background_image = Image.open("background_image_path.png")
    background_photo = ImageTk.PhotoImage(background_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=background_photo)

    image_paths = ["images/card_1.png", "images/card_2.png",
                   "images/card_3.png", "images/card_4.png",
                   "images/card_5.png", "images/card_6.png",
                   "images/card_7.png", "images/card_8.png",
                   "images/card_9.png"]

    images = []
    for path in image_paths:
        image = Image.open(path)
        photo = ImageTk.PhotoImage(image)
        images.append(photo)

    labels = []
    for i in range(len(my_deck)):
        label = tk.Label(root, image=images[my_deck[i]])
        label.pack(side="left")
        labels.append(label)

    play_game()  # 開始遊戲
    root.mainloop()


root = tk.Tk()

# 創建開始遊戲按鈕
play_button = tk.Button(root, text="開始遊戲", command=show_card_images)
play_button.pack()

# 載入圖像並創建圖像物件
image = Image.open("image/card_10.png")
photo = ImageTk.PhotoImage(image)

# 創建圖像標籤
label = tk.Label(root, image=photo)
label.pack()

root.mainloop()

# 建結束按鈕
quit_button = tk.Button(root, text="結束遊戲", command=show_card_images)
quit_button.pack()
