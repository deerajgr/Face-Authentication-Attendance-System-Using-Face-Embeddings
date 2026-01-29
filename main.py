import os
from register import register_user
from authenticate import authenticate_and_mark_attendance

def main_menu():
    while True:
        print("\n=== Face Authentication Attendance System ===")
        print("1. Register New User")
        print("2. Punch-In (Authenticate)")
        print("3. Punch-Out (Authenticate)")
        print("4. View Attendance Log")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            user_id = input("Enter User ID (your name): ").strip()
            if user_id:
                register_user(user_id)
            else:
                print("[ERROR] User ID cannot be empty.")
        
        elif choice == "2":
            authenticate_and_mark_attendance("Punch-In")
        
        elif choice == "3":
            authenticate_and_mark_attendance("Punch-Out")
        
        elif choice == "4":
            if os.path.exists("attendance_log.txt"):
                with open("attendance_log.txt", "r") as f:
                    print("\n--- Attendance Log ---")
                    print(f.read())
            else:
                print("[INFO] No attendance log yet.")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("[ERROR] Invalid choice. Try again.")

if __name__ == "__main__":
    main_menu()