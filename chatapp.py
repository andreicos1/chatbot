from tkinter import *
from tkinter.ttk import *
from omny_chatbot import chat

root = Tk()

root.title('Chat Bot')
root.geometry('400x500')
#Don't allow resizing in either direction
root.resizable(0, 0) 

#configure button style
sto = Style()
sto.configure('B1.TButton', foreground='black', background='#5FBFF9', font= ('Arial', 12, 'bold'),)
sto.configure('B2.TButton', foreground='black', background='green', font= ('Arial', 10),)

def send_msg():
	#get typed in message
	user_message = messageWindow.get()
	send_user_msg = "You: " + user_message +'\n'
	#insert it into the chat box, then delete it from the message entry window
	chatWindow.insert(END, send_user_msg)
	messageWindow.delete(0, END)

	send_bot_msg(user_message)

def send_bot_msg(user_message):
	#pass in the user message into the chatbot AI to generate a response
	bot_message = chat(user_message)
	send_bot_msg = "Cosmetic Bot: " + bot_message
	chatWindow.insert(END, send_bot_msg+'\n\n')

def clear_window():
	chatWindow.delete("1.0","end")

#Chat window
chatWindow = Text(root, bd=1, width=50, height=8)
chatWindow.place(x=6, y=6, height=485, width=380)
chatWindow.bind("<Key>", lambda e: "break")

#Message window
messageWindow = Entry(root, width=30)
messageWindow.place(x=12, y=460, height=30, width=265)

#send button
send = Button(root, text="Send", style='B1.TButton', command=send_msg,)
send.pack()
#make return key activate the button and place the button
root.bind('<Return>', lambda event=None: send.invoke())
send.place(x=285,y=460,height=30,width=95)
#button to clear windows
clear = Button(root, text="Clear window", command=clear_window, style='B2.TButton').place(x=285,y=424,height=30,width=95)
#start the bot and send a message
send_bot_msg("hi")
root.mainloop()
