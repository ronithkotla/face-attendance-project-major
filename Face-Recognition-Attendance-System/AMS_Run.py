import tkinter as tk
from tkinter import *
from tkinter import messagebox
import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
import time
import subprocess

# Window is our Main frame of system
window = tk.Tk()
window.title("FAMS-Face Recognition Based Attendance Management System")

window.geometry('1280x720')
window.configure(background='grey80')

# For clear textbox
def clear():
    txt.delete(first=0, last=22)

def clear1():
    txt2.delete(first=0, last=22)

def del_sc1():
    sc1.destroy()

def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry('300x100')
    sc1.title('Warning!!')
    sc1.configure(background='grey80')
    Label(sc1, text='Enrollment & Name required!!!', fg='black',
          bg='white', font=('times', 16)).pack()
    Button(sc1, text='OK', command=del_sc1, fg="black", bg="lawn green", width=9,
           height=1, activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

# Error screen2
def del_sc2():
    sc2.destroy()

def err_screen1():
    global sc2
    sc2 = tk.Tk()
    sc2.geometry('300x100')
    sc2.title('Warning!!')
    sc2.configure(background='grey80')
    Label(sc2, text='Please enter your subject name!!!', fg='black',
          bg='white', font=('times', 16)).pack()
    Button(sc2, text='OK', command=del_sc2, fg="black", bg="lawn green", width=9,
           height=1, activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

# For take images for datasets
def take_img():
    l1 = txt.get()
    l2 = txt2.get()
    l3 = txt3.get()  # Get password
    if l1 == '':
        err_screen()
    elif l2 == '':
        err_screen()
    elif l3 == '':  # Check if password is empty
        err_screen()
    else:
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml')
            Enrollment = txt.get()
            Name = txt2.get()
            Password = txt3.get()  # Store password
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder
                    cv2.imwrite("TrainingImage/ " + Name + "." + Enrollment + '.' + str(sampleNum) + ".jpg",
                                gray)
                    print("Images Saved for Enrollment :")
                    cv2.imshow('Frame', img)
                # wait for 100 miliseconds
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum > 70:
                    break

            cam.release()
            cv2.destroyAllWindows()
            ts = time.time()
            Date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            Time = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            row = [Enrollment, Name, Date, Time, Password]  # Include password in student details
            with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile, delimiter=',')
                writer.writerow(row)
                csvFile.close()
            res = "Images Saved for Enrollment : " + Enrollment + " Name : " + Name
            Notification.configure(
                text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
            Notification.place(x=250, y=100)
        except FileExistsError as F:
            f = 'Student Data already exists'
            Notification.configure(text=f, bg="Red", width=21)
            Notification.place(x=450, y=100)

# for choose subject and fill attendance
def subjectchoose():
    def Fillattendances():
        sub = tx.get()
        if sub == '':
            messagebox.showerror("Error", "Please enter subject name")
            return

        # Create Attendance directory if it doesn't exist
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")

        # Initialize camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        # Create or load attendance file
        attendance_file = os.path.join("Attendance", "Attendance.csv")
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Enrollment', 'Name', 'Date', 'Time', 'Subject'])

        # Load existing attendance data
        attendance_data = []
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                attendance_data = list(reader)

        # Load face recognition model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            recognizer.read("TrainingImageLabel/Trainner.yml")
        except:
            messagebox.showerror("Error", "Model not found, Please train model")
            cam.release()
            cv2.destroyAllWindows()
            return

        hcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(hcascadePath)
        
        # Load student details with error handling
        try:
            df = pd.read_csv("StudentDetails/StudentDetails.csv", on_bad_lines='skip')
            if df.empty:
                messagebox.showerror("Error", "No student details found")
                cam.release()
                cv2.destroyAllWindows()
                return
        except Exception as e:
            messagebox.showerror("Error", f"Error reading student details: {str(e)}")
            cam.release()
            cv2.destroyAllWindows()
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Enrollment', 'Name', 'Date', 'Time', 'Subject']
        attendance = pd.DataFrame(columns=col_names)

        # Get current date and time
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')

        # Process video frames
        while True:
            ret, im = cam.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image from camera")
                break

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                enrollment, conf = recognizer.predict(gray[y:y + h, x:x + w])
                enrollment_str = str(enrollment).lstrip('0').strip()
                df['Enrollment_stripped'] = df['Enrollment'].astype(str).str.lstrip('0').str.strip()
                match = df[df['Enrollment_stripped'] == enrollment_str]
                if conf < 50 and not match.empty:
                    name = match['Name'].values[0]
                    color = (0, 255, 0)
                    # Check if already present in today's attendance for this subject
                    already_present = False
                    if os.path.exists(attendance_file):
                        with open(attendance_file, 'r') as f:
                            reader = csv.reader(f)
                            next(reader)
                            for record in reader:
                                record_enrollment = str(record[0]).lstrip('0').strip()
                                if (len(record) >= 5 and 
                                    record_enrollment == enrollment_str and 
                                    record[4] == sub and 
                                    record[2] == date):
                                    already_present = True
                                    break
                    if not already_present:
                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([enrollment, name, date, time, sub])
                        print(f"Attendance marked for {enrollment} {name}")
                    label = str(enrollment) + " " + str(name)
                else:
                    name = "Unknown"
                    color = (225, 0, 0)
                    label = str(enrollment) + " " + str(name)
                cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
                cv2.putText(im, label, (x, y + h), font, 1, (255, 255, 255), 2)

            cv2.imshow('Attendance', im)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC
                break
            if cv2.getWindowProperty('Attendance', cv2.WND_PROP_VISIBLE) < 1:
                break

        # Save attendance
        if not attendance.empty:
            attendance.to_csv(attendance_file, mode='a', header=False, index=False)
            messagebox.showinfo("Success", "Attendance marked successfully!")

        # Clean up
        cam.release()
        cv2.destroyAllWindows()

        # Show attendance sheet
        root = tk.Tk()
        root.title("Attendance of " + sub)
        root.configure(background='grey80')
        
        # Create headers
        headers = ['Enrollment', 'Name', 'Date', 'Time', 'Subject']
        for c, header in enumerate(headers):
            label = tk.Label(root, text=header, width=15, height=2, fg="black", bg="grey",
                            font=('times', 12, 'bold'))
            label.grid(row=0, column=c, padx=5, pady=5)
        
        # Read and display attendance for this subject and date
        row_num = 1
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 5 and row[2] == date and row[4] == sub:
                    for c, value in enumerate(row):
                        label = tk.Label(root, text=str(value), width=15, height=1, fg="black",
                                        bg="white", font=('times', 10))
                        label.grid(row=row_num, column=c, padx=5, pady=2)
                    row_num += 1
        root.mainloop()

    # Create subject chooser window
    windo = tk.Tk()
    windo.title("Enter subject name...")
    windo.geometry('580x320')
    windo.configure(background='grey80')
    
    Notifica = tk.Label(windo, text="", bg="Green", fg="white", width=33,
                        height=2, font=('times', 15, 'bold'))

    def Attf():
        subprocess.Popen(
            r'explorer /select,"' + os.path.abspath('Attendance') + '"')

    attf = tk.Button(windo, text="Check Sheets", command=Attf, fg="white", bg="black",
                     width=12, height=1, activebackground="white", font=('times', 14, 'bold'))
    attf.place(x=430, y=255)

    sub = tk.Label(windo, text="Enter Subject : ", width=15, height=2,
                   fg="black", bg="grey", font=('times', 15, 'bold'))
    sub.place(x=30, y=100)

    tx = tk.Entry(windo, width=20, bg="white",
                  fg="black", font=('times', 23))
    tx.place(x=250, y=105)

    fill_a = tk.Button(windo, text="Fill Attendance", fg="white", command=Fillattendances, bg="SkyBlue1", width=20, height=2,
                       activebackground="white", font=('times', 15, 'bold'))
    fill_a.place(x=250, y=160)
    windo.mainloop()

def admin_panel():
    win = tk.Tk()
    win.title("LogIn")
    win.geometry('880x420')
    win.configure(background='grey80')

    def log_in():
        username = un_entr.get()
        password = pw_entr.get()

        if username == 'admin':
            if password == 'admin123':
                win.destroy()
                show_admin_options()
            else:
                valid = 'Incorrect ID or Password'
                Nt.configure(text=valid, bg="red", fg="white",
                             width=38, font=('times', 19, 'bold'))
                Nt.place(x=120, y=350)
        else:
            valid = 'Incorrect ID or Password'
            Nt.configure(text=valid, bg="red", fg="white",
                         width=38, font=('times', 19, 'bold'))
            Nt.place(x=120, y=350)

    def show_admin_options():
        admin_win = tk.Tk()
        admin_win.title("Admin Panel")
        admin_win.geometry('880x420')
        admin_win.configure(background='grey80')

        def view_student_details():
            import csv
            import tkinter
            root = tkinter.Tk()
            root.title("Student Details")
            root.configure(background='grey80')

            cs = os.path.join('StudentDetails', 'StudentDetails.csv')
            with open(cs, newline="") as file:
                reader = csv.reader(file)
                r = 0
                for col in reader:
                    c = 0
                    for row in col:
                        label = tkinter.Label(root, width=10, height=1, fg="black", font=('times', 15, ' bold '),
                                              bg="white", text=row, relief=tkinter.RIDGE)
                        label.grid(row=r, column=c)
                        c += 1
                    r += 1
            root.mainloop()

        def view_attendance_stats():
            try:
                # Read attendance data
                attendance_df = pd.read_csv('Attendance/Attendance.csv')
                student_df = pd.read_csv('StudentDetails/StudentDetails.csv')
                
                # Calculate total classes per subject
                total_classes = attendance_df.groupby('Subject').size()
                
                # Calculate attendance per student per subject
                attendance_stats = []
                for _, student in student_df.iterrows():
                    student_attendance = attendance_df[attendance_df['Enrollment'] == student['Enrollment']]
                    for subject in total_classes.index:
                        classes_attended = len(student_attendance[student_attendance['Subject'] == subject])
                        total = total_classes[subject]
                        percentage = (classes_attended / total * 100) if total > 0 else 0
                        attendance_stats.append({
                            'Name': student['Name'],
                            'Enrollment': student['Enrollment'],
                            'Subject': subject,
                            'Classes_Attended': classes_attended,
                            'Total_Classes': total,
                            'Attendance_Percentage': f"{percentage:.2f}%"
                        })
                
                # Create DataFrame and save to CSV
                stats_df = pd.DataFrame(attendance_stats)
                stats_df.to_csv('Attendance/Attendance_Statistics.csv', index=False)
                
                # Display the statistics
                root = tk.Tk()
                root.title("Attendance Statistics")
                root.configure(background='grey80')
                
                # Create headers
                headers = ['Name', 'Enrollment', 'Subject', 'Classes Attended', 'Total Classes', 'Attendance %']
                for c, header in enumerate(headers):
                    label = tk.Label(root, text=header, width=15, height=2, fg="black", bg="grey",
                                    font=('times', 12, 'bold'))
                    label.grid(row=0, column=c)
                
                # Display data
                for r, row in enumerate(stats_df.itertuples(), 1):
                    for c, value in enumerate(row[1:], 0):
                        label = tk.Label(root, text=str(value), width=15, height=1, fg="black",
                                        bg="white", font=('times', 10))
                        label.grid(row=r, column=c)
                
                root.mainloop()
                
            except Exception as e:
                error_win = tk.Tk()
                error_win.title("Error")
                error_win.geometry('300x100')
                error_win.configure(background='grey80')
                Label(error_win, text=f'Error: {str(e)}', fg='red',
                      bg='white', font=('times', 12)).pack()
                Button(error_win, text='OK', command=error_win.destroy, fg="black", bg="lawn green",
                       width=9, height=1, activebackground="Red",
                       font=('times', 12, 'bold')).place(x=90, y=50)
                error_win.mainloop()

        # Create buttons for admin options
        view_students_btn = tk.Button(admin_win, text="View Student Details", command=view_student_details,
                                     fg="black", bg="SkyBlue1", width=20, height=2,
                                     activebackground="white", font=('times', 15, 'bold'))
        view_students_btn.place(x=100, y=100)

        view_stats_btn = tk.Button(admin_win, text="View Attendance Statistics", command=view_attendance_stats,
                                  fg="black", bg="SkyBlue1", width=20, height=2,
                                  activebackground="white", font=('times', 15, 'bold'))
        view_stats_btn.place(x=400, y=100)

        admin_win.mainloop()

    Nt = tk.Label(win, text="Attendance filled Successfully", bg="Green", fg="white", width=40,
                  height=2, font=('times', 19, 'bold'))

    un = tk.Label(win, text="Enter username : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    un.place(x=30, y=50)

    pw = tk.Label(win, text="Enter password : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    pw.place(x=30, y=150)

    def c00():
        un_entr.delete(first=0, last=22)

    un_entr = tk.Entry(win, width=20, bg="white", fg="black",
                       font=('times', 23))
    un_entr.place(x=290, y=55)

    def c11():
        pw_entr.delete(first=0, last=22)

    pw_entr = tk.Entry(win, width=20, show="*", bg="white",
                       fg="black", font=('times', 23))
    pw_entr.place(x=290, y=155)

    c0 = tk.Button(win, text="Clear", command=c00, fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c0.place(x=690, y=55)

    c1 = tk.Button(win, text="Clear", command=c11, fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c1.place(x=690, y=155)

    Login = tk.Button(win, text="LogIn", fg="black", bg="SkyBlue1", width=20,
                      height=2,
                      activebackground="Red", command=log_in, font=('times', 15, ' bold '))
    Login.place(x=290, y=250)
    win.mainloop()

def student_login():
    win = tk.Tk()
    win.title("Student Login")
    win.geometry('880x420')
    win.configure(background='grey80')

    def log_in():
        enrollment = un_entr.get()
        password = pw_entr.get()

        if enrollment == "" or password == "":
            messagebox.showerror("Error", "Please enter both enrollment number and password")
            return

        try:
            # Read student details with proper error handling
            if not os.path.exists('StudentDetails/StudentDetails.csv'):
                messagebox.showerror("Error", "Student details file not found")
                return

            # Read CSV with error handling for malformed lines
            student_df = pd.read_csv('StudentDetails/StudentDetails.csv', on_bad_lines='skip')
            
            # Ensure the DataFrame has the required columns
            required_columns = ['Enrollment', 'Name', 'Password']
            if not all(col in student_df.columns for col in required_columns):
                messagebox.showerror("Error", "Student details file is not properly formatted")
                return

            # Check if enrollment exists
            entered_enrollment = enrollment.lstrip('0').strip()
            student_df['Enrollment_stripped'] = student_df['Enrollment'].astype(str).str.lstrip('0').str.strip()
            entered_password = password.strip()
            if entered_enrollment in student_df['Enrollment_stripped'].values:
                idx = student_df[student_df['Enrollment_stripped'] == entered_enrollment].index[0]
                stored_password = str(student_df.loc[idx, 'Password']).strip()
                if entered_password == stored_password:
                    win.destroy()
                    show_student_attendance(student_df.loc[idx, 'Enrollment'])
                else:
                    messagebox.showerror("Error", "Incorrect Password")
            else:
                messagebox.showerror("Error", "Enrollment not found")
        except Exception as e:
            messagebox.showerror("Error", f"Error during login: {str(e)}")

    def show_student_attendance(enrollment):
        try:
            # Read attendance data
            attendance_df = pd.read_csv('Attendance/Attendance.csv')
            student_df = pd.read_csv('StudentDetails/StudentDetails.csv')
            # Get student details
            student = student_df[student_df['Enrollment'].astype(str).str.lstrip('0').str.strip() == str(enrollment).lstrip('0').strip()].iloc[0]
            # Calculate total classes per subject
            total_classes = attendance_df.groupby('Subject').size()
            # Calculate attendance for this student
            student_attendance = attendance_df[attendance_df['Enrollment'].astype(str).str.lstrip('0').str.strip() == str(enrollment).lstrip('0').strip()]
            attendance_stats = []
            for subject in total_classes.index:
                classes_attended = len(student_attendance[student_attendance['Subject'] == subject])
                total = total_classes[subject]
                percentage = (classes_attended / total * 100) if total > 0 else 0
                attendance_stats.append({
                    'Subject': subject,
                    'Classes_Attended': classes_attended,
                    'Total_Classes': total,
                    'Attendance_Percentage': f"{percentage:.2f}%"
                })
            # Display the statistics
            root = tk.Tk()
            root.title(f"Attendance Statistics - {student['Name']}")
            root.configure(background='grey80')
            # Display student info using grid
            tk.Label(root, text=f"Name: {student['Name']}", font=('times', 14, 'bold'), bg='grey80').grid(row=0, column=0, columnspan=2, pady=(10,0))
            tk.Label(root, text=f"Enrollment: {enrollment}", font=('times', 14, 'bold'), bg='grey80').grid(row=1, column=0, columnspan=2, pady=(0,10))
            row_offset = 2
            # Create headers
            headers = ['Subject', 'Classes Attended', 'Total Classes', 'Attendance %']
            for c, header in enumerate(headers):
                label = tk.Label(root, text=header, width=15, height=2, fg="black", bg="grey",
                                font=('times', 12, 'bold'))
                label.grid(row=row_offset, column=c, padx=5, pady=5)
            # Display data
            for r, stat in enumerate(attendance_stats, 1):
                for c, (key, value) in enumerate(stat.items(), 0):
                    label = tk.Label(root, text=str(value), width=15, height=1, fg="black",
                                    bg="white", font=('times', 10))
                    label.grid(row=row_offset + r, column=c, padx=5, pady=2)
            root.mainloop()
        except Exception as e:
            error_win = tk.Tk()
            error_win.title("Error")
            error_win.geometry('300x100')
            error_win.configure(background='grey80')
            Label(error_win, text=f'Error: {str(e)}', fg='red',
                  bg='white', font=('times', 12)).pack()
            Button(error_win, text='OK', command=error_win.destroy, fg="black", bg="lawn green",
                   width=9, height=1, activebackground="Red",
                   font=('times', 12, 'bold')).place(x=90, y=50)
            error_win.mainloop()

    Nt = tk.Label(win, text="", bg="Green", fg="white", width=40,
                  height=2, font=('times', 19, 'bold'))

    un = tk.Label(win, text="Enter Enrollment : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    un.place(x=30, y=50)

    pw = tk.Label(win, text="Enter Password : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    pw.place(x=30, y=150)

    def c00():
        un_entr.delete(first=0, last=22)

    un_entr = tk.Entry(win, width=20, bg="white", fg="black",
                       font=('times', 23))
    un_entr.place(x=290, y=55)

    def c11():
        pw_entr.delete(first=0, last=22)

    pw_entr = tk.Entry(win, width=20, show="*", bg="white",
                       fg="black", font=('times', 23))
    pw_entr.place(x=290, y=155)

    c0 = tk.Button(win, text="Clear", command=c00, fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c0.place(x=690, y=55)

    c1 = tk.Button(win, text="Clear", command=c11, fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c1.place(x=690, y=155)

    Login = tk.Button(win, text="LogIn", fg="black", bg="SkyBlue1", width=20,
                      height=2,
                      activebackground="Red", command=log_in, font=('times', 15, ' bold '))
    Login.place(x=290, y=250)
    win.mainloop()

# For train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces, Id
        faces, Id = getImagesAndLabels("TrainingImage")
    except Exception as e:
        l = 'please make "TrainingImage" folder & put Images'
        Notification.configure(text=l, bg="SpringGreen3",
                               width=50, font=('times', 18, 'bold'))
        Notification.place(x=350, y=100)

    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("TrainingImageLabel/Trainner.yml")
    except Exception as e:
        q = 'Please make "TrainingImageLabel" folder'
        Notification.configure(text=q, bg="SpringGreen3",
                               width=50, font=('times', 18, 'bold'))
        Notification.place(x=350, y=100)

    res = "Model Trained"
    Notification.configure(text=res, bg="olive drab",
                           width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=100)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System", bg="black", fg="white", width=50,
                   height=3, font=('times', 30, ' bold '))

message.place(x=80, y=20)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,
                        height=3, font=('times', 17))

lbl = tk.Label(window, text="Enter Enrollment : ", width=20, height=2,
               fg="black", bg="grey", font=('times', 15, 'bold'))
lbl.place(x=200, y=200)

def testVal(inStr, acttyp):
    if acttyp == '1':  # insert
        if not inStr.isdigit():
            return False
    return True

txt = tk.Entry(window, validate="key", width=20, bg="white",
               fg="black", font=('times', 25))
txt['validatecommand'] = (txt.register(testVal), '%P', '%d')
txt.place(x=550, y=210)

lbl2 = tk.Label(window, text="Enter Name : ", width=20, fg="black",
                bg="grey", height=2, font=('times', 15, ' bold '))
lbl2.place(x=200, y=300)

txt2 = tk.Entry(window, width=20, bg="white",
                fg="black", font=('times', 25))
txt2.place(x=550, y=310)

lbl3 = tk.Label(window, text="Enter Password : ", width=20, fg="black",
                bg="grey", height=2, font=('times', 15, ' bold '))
lbl3.place(x=200, y=400)

txt3 = tk.Entry(window, width=20, show="*", bg="white",
                fg="black", font=('times', 25))
txt3.place(x=550, y=410)

def clear2():
    txt3.delete(first=0, last=22)

clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="white", bg="black",
                         width=10, height=1, activebackground="white", font=('times', 15, ' bold '))
clearButton2.place(x=950, y=410)

AP = tk.Button(window, text="Check Registered students", command=admin_panel, fg="black",
               bg="SkyBlue1", width=19, height=1, activebackground="white", font=('times', 15, ' bold '))
AP.place(x=990, y=410)

takeImg = tk.Button(window, text="Take Images", command=take_img, fg="black", bg="SkyBlue1",
                    width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=90, y=500)

trainImg = tk.Button(window, text="Train Images", fg="black", command=trainimg, bg="SkyBlue1",
                     width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=390, y=500)

FA = tk.Button(window, text="Automatic Attendance", fg="black", command=subjectchoose,
               bg="SkyBlue1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
FA.place(x=690, y=500)

# Add Student Login button to main window
student_login_btn = tk.Button(window, text="Student Login", command=student_login, fg="black",
                             bg="SkyBlue1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
student_login_btn.place(x=990, y=500)

window.mainloop()
