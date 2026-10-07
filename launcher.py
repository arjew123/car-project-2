import os
import sys
import subprocess
import webbrowser
import time

def main():
    # 1. Lock into the correct directory
    if getattr(sys, 'frozen', False):
        current_dir = os.path.dirname(sys.executable)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
    os.chdir(current_dir)
    
    if not os.path.exists("package.json"):
        print("ERROR: package.json not found! Place this .exe in your project folder.")
        input("Press Enter to exit...")
        return

    if not os.path.exists("node_modules"):
        print("Installing dependencies for the first time...")
        subprocess.run("npm install --legacy-peer-deps", shell=True)

    server_process = None

    # 2. Interactive Control Menu Loop
    while True:
        # Check if the server is currently running
        is_running = server_process is not None and server_process.poll() is None
        
        print("\n" + "="*35)
        print(" AutoValuate India - Control Panel")
        print("="*35)
        print(f" Status: {'[RUNNING]' if is_running else '[STOPPED]'}")
        print("\n 1. Start Server")
        print(" 2. Stop Server")
        print(" 3. Open App in Browser")
        print(" 4. Exit")
        print("="*35)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == '1':
            if is_running:
                print("\n-> Server is already running.")
            else:
                print("\n-> Booting up server...")
                server_process = subprocess.Popen("npm run dev", shell=True)
                time.sleep(4) # Give Vite time to start
                print("-> Server started.")
                
        elif choice == '2':
            if is_running:
                print("\n-> Stopping server...")
                # Force kill the npm process and all its child Node.js processes
                subprocess.run(f"taskkill /F /T /PID {server_process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                server_process = None
                print("-> Server completely stopped.")
            else:
                print("\n-> Server is not currently running.")
                
        elif choice == '3':
            print("\n-> Opening browser...")
            webbrowser.open("http://localhost:3000")
            
        elif choice == '4':
            if is_running:
                print("\n-> Stopping server before exiting...")
                subprocess.run(f"taskkill /F /T /PID {server_process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            break
            
        else:
            print("\n-> Invalid input. Please type 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()