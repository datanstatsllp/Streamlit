import streamlit as st
import pandas as pd
from PIL import Image
import os
import face_recognition
import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp
from streamlit_navigation_bar import st_navbar

st.set_page_config(layout="wide", page_title="DataNStats: Visual AI Solution", page_icon = 'camera')

############################################ Home Page Function ####################################
# Define individual page functions
def home_page():
    page_bg_img = """
    <style>

    [data-testid="stHeader"] {
        background: rgba(0, 255, 0, 0);  /* Transparent header */
    }
    </style>
    """
    
    # Apply the CSS
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # Your Streamlit app content
    st.title("Welcome to Servilance System")
    st.write("This Visual AIapp is used for real-time monitoring and hazard detection.")

############################################ Laptop Camera Monitoring Page ####################################    
# Define individual page functions
def monitoring_page():
    
    st.header("Visual AI: Society Attendance Solution", divider=True)
    
    # Ensure unmatched faces directory exists
    unmatched_faces_dir = "unmatched_faces"
    os.makedirs(unmatched_faces_dir, exist_ok=True)

    # Load employee details from emp.csv
    emp_csv_path = "user_data.xlsx"
    emp_details = pd.DataFrame()

    if os.path.exists(emp_csv_path):
        emp_details = pd.read_excel(emp_csv_path, sheet_name=0, index_col=None)
    else:
        st.error("user_data.csv file not found. Please ensure it is in the correct directory.")

    col1, col2 = st.columns([2, 3])

    with col1:
        # Known face encodings and names
        known_face_encodings = [] 
        known_face_names = emp_details["Name"].tolist()
        
        for index, row in emp_details.iterrows():
            name = row["Name"]
            image_path = f"D:/Product/Facial Recognizatio/img/{name}.jpeg"
            if os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
            else:
                st.warning(f"Image for {name} not found at {image_path}")
        
        # Attendance logging file setup
        current_date = datetime.now().strftime("%d-%m-%y")
        attendance_file = f"{current_date}_attendance.csv"
        
        if os.path.exists(attendance_file):
            attendance_log = pd.read_csv(attendance_file, index_col=None)
        else:
            attendance_log = pd.DataFrame(columns=["Name", "Flat", "Wing", "Time"], index=None)
            attendance_log.to_csv(attendance_file, index=False)
        
        # Capture image through Streamlit camera
        img_data = st.camera_input("**Take a picture for attendance**")
        
        if img_data:
            # Convert captured image to OpenCV format
            img_array = np.array(bytearray(img_data.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            # Recognize faces in the captured image
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
            for i, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if any(matches):
                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index]
                    emp_info = emp_details[emp_details["Name"] == name].iloc[0].to_dict()
        
                    # Log attendance if not already logged for today
                    if not attendance_log[
                        (attendance_log["Name"] == name) &
                        (attendance_log["Time"] == datetime.now().strftime("%H:%M:%S"))
                    ].any().any():
                        new_entry = pd.DataFrame({
                            "Name": [name],
                            "Flat No": [emp_info["Flat"]],
                            "Wing": [emp_info["Wing"]],
                            "Time": [datetime.now().strftime("%H:%M:%S")]
                        })
                        attendance_log = pd.concat([attendance_log, new_entry], ignore_index=True)
                        attendance_log.to_csv(attendance_file, index=False)
                else:
                    # Save unmatched face
                    top, right, bottom, left = face_locations[i]
                    unmatched_face = rgb_img[top:bottom, left:right]
                    unmatched_face_image = cv2.cvtColor(unmatched_face, cv2.COLOR_RGB2BGR)
                    unmatched_face_filename = os.path.join(unmatched_faces_dir, f"unmatched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(unmatched_face_filename, unmatched_face_image)
                    st.warning(f"Unmatched face detected and saved as {unmatched_face_filename}")

    with col2:
        # Define a function to dynamically update time
        def get_current_time():
            return datetime.now().strftime("%H:%M:%S")
        
        with st.container():
            # Metrics
            current_time = get_current_time()
            
            unique_visitors = attendance_log["Name"].nunique()
            total_visitors = attendance_log["Name"].count()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Date", current_date)
            col2.metric("Total Visitors", total_visitors, delta_color="inverse")
            col3.metric("Unique Visitors", unique_visitors, delta_color="inverse")
            col4.metric("Current Time", current_time)
        
        with st.container():
            # Display the attendance log in a table format
            st.write("**Attendance Log:**")
            st.write(attendance_log)

############################################ User Form Page Function ####################################
def user_page():
    
    # Directory to save images
    IMAGE_SAVE_PATH = r"D:\Product\Facial Recognizatio\img"

    # Ensure the directory exists
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

    # Function to save data into Excel
    def save_to_excel(data, filename='user_data.xlsx'):
        # Check if the file already exists
        try:
            # Read the existing data from the file (if it exists)
            df = pd.read_excel(filename)
        except FileNotFoundError:
            # If the file doesn't exist, create a new DataFrame
            df = pd.DataFrame(columns=["Name", "Gender", "Wing","Flat", "Pic"])

        # Convert the new data to a DataFrame
        new_row = pd.DataFrame([data])  # Pass a list of dictionaries

        # Append the new data using pd.concat
        df = pd.concat([df, new_row], ignore_index=True)

        # Save the updated DataFrame back to Excel
        df.to_excel(filename, index=False)
    
    with st.container():
        # Streamlit interface
        st.header("Add User Form", divider=True)
        
        # Input fields
        name = st.text_input('Enter your name:')
        gender = st.selectbox('Select your gender:', ['Male', 'Female', 'Other'], index=None, placeholder="Select user gender...")
        wing = st.selectbox('Select the Wing: ', ('A-Wing', 'B-Wing', 'C-Wing'), index=None, placeholder="Select Wing Name...")
        flat = st.text_input('Enter your Flat No:')
        pic = st.file_uploader("Upload your picture", type=["jpg", "jpeg", "png"])
    
        # When user clicks on "Submit"
        if st.button('Submit'):
            if name and flat and gender and wing and pic:
                try:
                    # Read the uploaded image
                    image = Image.open(pic)
    
                    # Save the image to the specified directory
                    image_filename = os.path.join(IMAGE_SAVE_PATH, f"{name}.jpeg")
                    image.save(image_filename)
    
                    # Prepare data dictionary
                    data = {
                        "Name": name,
                        "Gender": gender,
                        "Wing": wing,
                        "Flat": flat,
                        "Pic": image_filename  # Store the path to the image file
                    }
    
                    # Save data to Excel file
                    save_to_excel(data)
                    st.success("Data saved successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Please fill all fields.")

############################################ Contact Page Function ####################################
def contact_page():
    st.title("Contact Page")
    st.write("Welcome to the Contact Page!")
    st.write("Feel free to reach out to us: info@datanstats.com")
    
############################################ Hazard Page Function ####################################
def hazard_page():

    # Mediapipe Pose Initialization
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    # Function to check if someone fell down
    def check_fall(landmarks):
        if landmarks:
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            ankle_left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
            ankle_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
    
            avg_ankle_y = (ankle_left_y + ankle_right_y) / 2
            if nose_y > avg_ankle_y - 0.1:  # Adjust threshold as necessary
                return True
        return False
    
    # Streamlit App
    st.header("Hazard Detection App", divider=True)
    
    # Initialize DataFrame for logging falls
    fall_log = pd.DataFrame(columns=["Date", "Time"])
    
    # Layout: Split into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Live Camera Feed")
        stframe = st.empty()  # Placeholder for video frame
    
    with col2:
        st.subheader("Fall Detection Log")
        fall_table = st.empty()  # Placeholder for fall detection log table
    
    # Video Stream Handling
    run_detection = st.checkbox("Start Detection")
    
    if run_detection:
        # OpenCV Video Capture
        cap = cv2.VideoCapture(0)  # Default webcam (change to video file if needed)
    
        # Reduce camera resolution
        width, height = 640, 480  # Set desired resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Error accessing the camera or video file.")
                break
    
            # Convert to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
    
            # Draw pose landmarks and check for fall
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                if check_fall(results.pose_landmarks.landmark):
                    cv2.putText(
                        frame,
                        "Fall Detected!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
    
                    # Log the fall detection
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    
                    # Create a new DataFrame for the detected fall
                    new_log = pd.DataFrame({"Date": [date_str], "Time": [time_str]})
                    
                    # Concatenate the new log entry
                    fall_log = pd.concat([fall_log, new_log], ignore_index=True)
    
                    # Update fall log table in Streamlit
                    fall_table.dataframe(fall_log)
    
            # Display the video frame in the first column
            stframe.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True
            )
    
        cap.release()

############################################ Contact Page Function ####################################
def usrlist_page():
    # File name
    file_name = "user_data.xlsx"
    
    # Streamlit App
    st.header("Visual AI Solution User List", divider=True)
    
    # Check if the file exists
    if os.path.exists(file_name):
        #st.success(f"File '{file_name}' found!")
    
        # Read and display the Excel file
        try:
            df = pd.read_excel(file_name)
            st.write("File Content:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the Excel file: {e}")
    else:
        st.warning(f"File '{file_name}' does not exist in the current directory.")
    
############################################ Navigation Bar in Sidebar ####################################
page = st_navbar(["Home", "Monitoring", "Hazard", "User", "User List", "Contact","Log Out"])

# Render the selected page
if page == "Home":
    home_page()
elif page == "User":
    user_page()
elif page == "Monitoring":
    monitoring_page()
elif page == "Contact":
    contact_page()
elif page == "Hazard":
    hazard_page()
elif page == "User List":
    usrlist_page()


