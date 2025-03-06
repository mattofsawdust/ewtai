"""
Simple Streamlit app to test basic file creation and download functionality.
This is a minimal test case to verify that file operations work in the environment.
"""

import streamlit as st
import os
from datetime import datetime

# Set page title
st.title("File Creation and Download Test")
st.write("This is a minimal app to test file creation and download functionality")

# Create a simple layout
st.markdown("### Test Controls")

# Create a simple test file
if st.button("Create Test File"):
    try:
        # Create results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        st.write(f"Created/confirmed directory at: {results_dir}")
        
        # Create a test file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_file_path = os.path.join(results_dir, f"test_file_{timestamp}.txt")
        
        # Write content to the file
        with open(test_file_path, 'w') as f:
            f.write(f"Test file created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"This is a simple test file to verify download functionality.\n")
            f.write(f"Random data: 12345-ABCDE-67890\n")
        
        st.success(f"✅ File created successfully at: {test_file_path}")
        
        # Verify the file exists
        if os.path.exists(test_file_path):
            st.write("File exists after creation.")
            st.write(f"File size: {os.path.getsize(test_file_path)} bytes")
            
            # Read the file content
            with open(test_file_path, 'r') as f:
                content = f.read()
            
            # Display file content
            st.text_area("File Content:", value=content, height=100)
            
            # Try to create a download button with the file
            with open(test_file_path, 'rb') as f:
                file_data = f.read()
            
            st.download_button(
                label="Download Test File",
                data=file_data,
                file_name=os.path.basename(test_file_path),
                mime="text/plain"
            )
            
            st.write("If the download button doesn't work, try accessing the file directly:")
            st.code(f"open {test_file_path}")
        else:
            st.error(f"File does not exist after creation attempt: {test_file_path}")
    
    except Exception as e:
        st.error(f"Error during test: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Display environment information
st.markdown("### Environment Information")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Python executable: {os.path.dirname(os.path.dirname(os.__file__))}")
st.write(f"Streamlit version: {st.__version__}")

# Try to list files in the directory
st.markdown("### Files in current directory")
try:
    files = os.listdir(os.getcwd())
    st.write(f"Found {len(files)} files/directories:")
    for file in files[:10]:  # Show only first 10 to avoid clutter
        st.write(f"- {file}")
    if len(files) > 10:
        st.write(f"... and {len(files) - 10} more")
except Exception as e:
    st.error(f"Error listing files: {str(e)}")

st.markdown("---")
st.markdown("Test app created to diagnose file download issues")