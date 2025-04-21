import streamlit as st
import subprocess
import threading
import os
import signal

# Global process reference
process = None

def run_script(script_path):
    global process

    # Use subprocess to run the external script
    def launch():
        global process
        process = subprocess.Popen(['python', script_path])

        # Wait for the process to finish
        process.wait()
        
        # Once process finishes, rerun Streamlit app to return to home
        st.rerun()

    # Launch the process in a new thread
    thread = threading.Thread(target=launch)
    thread.start()

def stop_script():
    global process
    if process and process.poll() is None:  # Check if process is running
        os.kill(process.pid, signal.SIGTERM)
        process = None
