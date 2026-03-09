from register_user import register_user
from login import login
from database import show_all_users, delete_user, get_audit_log, delete_all_users, is_database_empty, get_user_count

def main():
    while True:
        print("\n" + "="*60)
        print("     FACE LOGIN SYSTEM")
        print("="*60)
        
        if is_database_empty():
            print("     DATABASE: EMPTY - No users")
        else:
            print(f"     DATABASE: {get_user_count()} user(s)")
        
        print("="*60)
        print("  1. Register New User")
        print("  2. Login")
        print("  3. View All Users")
        print("  4. Delete User")
        print("  5. View Audit Log")
        print("  6. DELETE ALL USERS")
        print("  7. Exit")
        print("="*60)

        choice = input("Enter choice (1-7): ").strip()

        if choice == "1":
            register_user()
        elif choice == "2":
            login()
        elif choice == "3":
            show_all_users()
        elif choice == "4":
            name = input("Enter name to delete: ").strip().lower()
            delete_user(name)
        elif choice == "5":
            logs = get_audit_log(10)
            print("\n=== Recent Logins ===")
            for log in logs:
                status = "✅" if log['success'] else "❌"
                print(f"{log['timestamp']} {status} {log['employee']} - {log['action']}")
        elif choice == "6":
            confirm = input("Type 'DELETE ALL' to confirm: ")
            if confirm == "DELETE ALL":
                delete_all_users()
        elif choice == "7":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()