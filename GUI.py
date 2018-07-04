from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import run

if __name__ == "__main__":
    root = Tk()  
    root.title('Face Recognition')
    root.geometry('400x500') 
    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #choose 
    def choose():
        global filename
        global File
        File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
        filename = ImageTk.PhotoImage(Image.open(File))
        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(0,0,anchor='nw',image=filename)
        
    var = StringVar()
    #predict   
    def predict():
        tp = run.pre(File)
        print(tp)
        
        var.set('Top-5:\n'+str(int(tp[0][0]))+': '+str(round(tp[1][0],3)*100)+'%\n'+
                 str(int(tp[0][1]))+': '+str(round(tp[1][1],3)*100)+'%\n'+
                 str(int(tp[0][2]))+': '+str(round(tp[1][2],3)*100)+'%\n'+
                 str(int(tp[0][3]))+': '+str(round(tp[1][3],3)*100)+'%\n'+
                 str(int(tp[0][4]))+': '+str(round(tp[1][4],3)*100)+'%\n')###output機率最高的人###

    Button(root,text='choose',command=choose).pack()
    Button(root,text='predict',command=predict).pack()
    label_text = Label(root, textvariable=var)
    label_text.pack()     
    root.mainloop()
